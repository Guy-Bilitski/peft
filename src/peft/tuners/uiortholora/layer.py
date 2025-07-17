from __future__ import annotations
from typing import Dict, List, Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import dequantize_module_weight
from transformers.pytorch_utils import Conv1D

__all__ = ["UIOrthoLoRALayer", "Linear"]

class IdentityWithTranspose(nn.Identity):
    @property
    def T(self):
        return self

    def __matmul__(self, other):
        return other  # Identity @ other

    def __rmatmul__(self, other):
        return other  # other @ Identity


class UIOrthoLoRALayer(BaseTunerLayer):
    adapter_layer_names = ("uiortholora_sigma", "uiortholora_D", "uiortholora_E")
    other_param_names = ("uiortholora_alpha", "uiortholora_dropout", "rank", "scaling_factor", "enforce_sv_positive")

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
         # ---- handle Linear vs Conv1D ----------------------------------
        if isinstance(base_layer, Conv1D):
            # Conv1D stores weight with shape (in_dim, out_dim)
            self.in_features  = base_layer.weight.shape[0]
            self.out_features = base_layer.weight.shape[1]
        else:                              # nn.Linear / 8-bit / 4-bit
            self.in_features  = base_layer.in_features
            self.out_features = base_layer.out_features

        self.uiortholora_sigma = nn.ParameterDict()
        self.uiortholora_D = nn.ParameterDict()
        self.uiortholora_E = nn.ParameterDict()
        self.uiortholora_left_unitary = nn.ParameterDict()
        self.uiortholora_right_unitary = nn.ParameterDict()
        self.uiortholora_dropout = nn.ModuleDict()
        self.device = self.base_layer.weight.device
        self._meta: Dict[str, dict] = {}
        self.kwargs = kwargs
        self.num_svalues_to_adapt = kwargs.pop("num_svalues_to_adapt")
        self.num_svectors_to_adapt = kwargs.pop("num_svectors_to_adapt")

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def update_layer(
        self,
        adapter_name: str,
        *,
        scaling_factor: float = 1.0,
        enforce_sv_positive: bool = False,
        uiortholora_dropout: float = 0.0,
        initial_scaler: Optional[float] = 1e-1,
        initial_sigma: Optional[float] = 1e-1,
        **kwargs,
    ):
            
        if adapter_name in self.uiortholora_sigma.keys():
            return

        base_w = self.get_base_layer().weight

        try:
            base_w = dequantize_module_weight(self.get_base_layer())
        except (TypeError, AttributeError):
            pass

        if base_w.dim() > 2:
            base_w = base_w.view(base_w.size(0), -1)
        elif base_w.dim() == 1:
            base_w = base_w.view(1, -1)

        if self.fan_in_fan_out:
            base_w = base_w.T  # transpose before doing SVD

        rank = min(self.in_features, self.out_features)
        # rank_to_preserve = rank - self.num_svectors_to_adapt
        major_component_size = rank - self.num_svalues_to_adapt
        medium_component_size = rank - self.num_svectors_to_adapt

        # Compute SVD and slice the smallest singular vectors
        U, S, Vt = torch.linalg.svd(base_w.float(), full_matrices=False)
        print(f"Calculated SVD")
        if self.num_svalues_to_adapt == 0:
            ids = torch.argsort(S)[:self.num_svalues_to_adapt]
            U, Vt = U[:, ids], Vt[ids, :]

        # Major component
        U1 = U[:, :major_component_size].detach()
        S1 = S[:major_component_size].detach()
        Vt1 = Vt[:major_component_size, :].detach()
        with torch.no_grad():
            major_component = (U1 * S1) @ Vt1
        self.register_buffer(f"{adapter_name}_major_component", major_component, persistent=True)

        # Medium component between svalues and svectors
        self.register_buffer(f"{adapter_name}_U2", U[:, major_component_size:medium_component_size].detach(), persistent=True)
        self.register_buffer(f"{adapter_name}_S2", S[major_component_size:medium_component_size].detach(), persistent=True)
        self.register_buffer(f"{adapter_name}_Vt2", Vt[major_component_size:medium_component_size, :].detach(), persistent=True)

        # Small component
        self.register_buffer(f"{adapter_name}_U3", U[:, medium_component_size:].detach(), persistent=True)
        self.register_buffer(f"{adapter_name}_S3", S[medium_component_size:].detach(), persistent=True)
        self.register_buffer(f"{adapter_name}_Vt3", Vt[medium_component_size:, :].detach(), persistent=True)

        # Initialize parameters with provided values or defaults
        self.uiortholora_sigma[adapter_name] = nn.Parameter(torch.full((self.num_svalues_to_adapt,), initial_sigma, dtype=torch.float))

        # Initialize D and E with provided scaler or default of 1
        self.uiortholora_D[adapter_name] = nn.Parameter(torch.full((self.in_features,), initial_scaler, dtype=torch.float))
        self.uiortholora_E[adapter_name] = nn.Parameter(torch.full((self.out_features,), initial_scaler, dtype=torch.float))

        if self.num_svectors_to_adapt == 0:
            left_orthogonal = IdentityWithTranspose()
            right_orthogonal = IdentityWithTranspose()

        else:
            left_orthogonal = torch.nn.utils.parametrizations.orthogonal(
                nn.Linear(self.num_svectors_to_adapt, self.num_svectors_to_adapt, bias=False)
            )
            right_orthogonal = torch.nn.utils.parametrizations.orthogonal(
                nn.Linear(self.num_svectors_to_adapt, self.num_svectors_to_adapt, bias=False)
            )
        
        self.uiortholora_left_unitary[adapter_name] = left_orthogonal
        self.uiortholora_right_unitary[adapter_name] = right_orthogonal

        self._meta[adapter_name] = dict(sf=scaling_factor, pos=enforce_sv_positive)

        # Add dropout
        self.uiortholora_dropout.update(nn.ModuleDict({
            adapter_name: nn.Dropout(uiortholora_dropout) if uiortholora_dropout > 0.0 else nn.Identity()
        }))

        # Move newly added parameters to the base layer's device
        self._move_adapter_to_device_of_base_layer(adapter_name)

        # Activate the adapter
        self.set_adapter(self.active_adapters)


    def get_base_layer(self) -> nn.Module:
        return self.base_layer if not hasattr(self.base_layer, "get_base_layer") else self.base_layer.get_base_layer()

class Linear(nn.Linear, UIOrthoLoRALayer):
    def __init__(
        self,
        base_layer: nn.Linear,
        adapter_name: str,
        *,
        uiortholora_alpha: float = 1.0,
        uiortholora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        **kwargs,
    ) -> None:
        # Initialize nn.Linear with the base layer's parameters
        super(nn.Linear, self).__init__()
        # Initialize UIOrthoLoRALayer
        UIOrthoLoRALayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            scaling_factor=kwargs.pop("scaling_factor"),
            enforce_sv_positive=kwargs.pop("enforce_sv_positive"),
            uiortholora_alpha=uiortholora_alpha,
            uiortholora_dropout=uiortholora_dropout,
            initial_scaler=kwargs.pop("initial_scaler"),
            initial_sigma=kwargs.pop("initial_sigma"))

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        """
        Return the effective weight matrix of the adapter: ΔW = E * Q * U * Σ * (V * P).T * D

        This is a proper, differentiable version safe for inspection or merging.
        """
        diag = self.uiortholora_sigma[adapter]
        # if self._meta[adapter]["pos"]:
        #     diag = torch.relu(diag.clone())

        U = getattr(self, f"{adapter}_U")
        Vt = getattr(self, f"{adapter}_Vt")

        D = self.uiortholora_D[adapter]
        E = self.uiortholora_E[adapter]
        left_unitary = self._calc_left_unitary(self.uiortholora_left_unitary[adapter], self.in_features)
        right_unitary = self._calc_right_unitary(self.uiortholora_right_unitary[adapter], self.out_features)
        # Broadcast D and E
        VtD = Vt * D.unsqueeze(0).clone()
        EU = U * E.unsqueeze(1).clone()

        orthogonal_size = min(self.in_features, self.out_features)
        Σ = self._calc_sigma(diag, orthogonal_size)

        core = EU @ left_unitary @ Σ @ right_unitary.T @ VtD
        return self._meta[adapter]["sf"] * core.to(self.get_base_layer().weight.dtype)



    def merge(self, *, safe_merge: bool = False, adapter_names: Optional[List[str]] = None):
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.uiortholora_sigma.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()

                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self):
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.uiortholora_sigma.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def _calc_tuner_internal(self, adapter: str):
        major_component = getattr(self, f"{adapter}_major_component").detach().clone()
        U2 = getattr(self, f"{adapter}_U2")
        Vt2 = getattr(self, f"{adapter}_Vt2")
        S2 = getattr(self, f"{adapter}_S2")
        U3 = getattr(self, f"{adapter}_U3")
        Vt3 = getattr(self, f"{adapter}_Vt3")
        S3 = getattr(self, f"{adapter}_S3")

        if self.num_svectors_to_adapt > 0:
            new_U3 = U3 @ self.uiortholora_left_unitary[adapter].weight
            new_Vt3 = (Vt3.T @ self.uiortholora_right_unitary[adapter].weight).T
        else:
            new_U3 = U3
            new_Vt3 = Vt3

        major_component.addmm_(U2 * S2, Vt2, beta=1.0, alpha=1.0)
        major_component.addmm_(new_U3 * S3, new_Vt3, beta=1.0, alpha=1.0)
        return major_component



    def forward(self, x: torch.Tensor, *args, **kwargs):
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return self.base_layer(x, *args, **kwargs)

        if self.merged:
            return self.base_layer(x, *args, **kwargs)

        result = self.base_layer(x, *args, **kwargs)

        for name in self.active_adapters:
            if name not in self.uiortholora_sigma.keys():
                continue

            diag = self.uiortholora_sigma[name]
            if self._meta[name]["pos"]:
                diag = torch.relu(diag)

            # U = getattr(self, f"{name}_U")               # (out, r)
            # Vt = getattr(self, f"{name}_Vt")               # (r, in)
            D = self.uiortholora_D[name]                   # (in,)
            E = self.uiortholora_E[name]
            # orthogonal_size = min(self.in_features, self.out_features)
            # left_unitary = self._calc_left_unitary(self.uiortholora_left_unitary[name], orthogonal_size)
            # right_unitary = self._calc_right_unitary(self.uiortholora_right_unitary[name], orthogonal_size)
            

            x_casted = x.to(diag.dtype)
            svd_tuner = self._calc_tuner_internal(name)
            x_proj = F.linear(self.uiortholora_dropout[name](x_casted), svd_tuner * D.unsqueeze(0))

            delta = self._meta[name]["sf"] * x_proj * E.view(1,1,-1)
            result = result + delta
            
            # x_proj = F.linear(self.uiortholora_dropout[name](x_casted), (right_unitary.T @ Vt) * D.unsqueeze(0))
            # sigma = self._calc_sigma(diag, orthogonal_size)
            # x_proj = x_proj @ sigma
            # delta = F.linear(x_proj, (U @ left_unitary) * E.unsqueeze(1))

            # result = result + self._meta[name]["sf"] * delta

        return result


    def get_base_layer(self):
        return self.base_layer

    def __repr__(self):
        return f"UIOrthoLoRALayer({self.get_base_layer().__repr__()})"
    
    def _calc_left_unitary(self, left_unitary, left_size):
        if self.num_svectors_to_adapt == 0:
            return left_unitary

        rank_to_preserve = left_size - self.num_svectors_to_adapt
        return self._build_projection_matrix(left_unitary.weight, left_size, rank_to_preserve)
    
    def _calc_right_unitary(self, right_unitary, right_size):
        if self.num_svectors_to_adapt == 0:
            return right_unitary

        rank_to_preserve = right_size - self.num_svectors_to_adapt
        return self._build_projection_matrix(right_unitary.weight, right_size, rank_to_preserve)
    
    def _build_projection_matrix(self, projection_matrix, size, rank_to_preserve):
        upper_matrix = torch.eye(rank_to_preserve, rank_to_preserve, device = self.get_base_layer().weight.device)
        upper_matrix = torch.cat((upper_matrix, torch.zeros(rank_to_preserve, size - rank_to_preserve, device = self.get_base_layer().weight.device)), dim=1)
        down_matrix = torch.cat((torch.zeros(size - rank_to_preserve, rank_to_preserve, device = self.get_base_layer().weight.device), projection_matrix), dim=1)
        return torch.cat((upper_matrix, down_matrix), dim=0)
    
    def _calc_sigma(self, diag_values, orthogonal_size):
        device = self.get_base_layer().weight.device
        # max_rank = min(left_size, right_size)
        not_trainable_part_size = orthogonal_size - self.num_svalues_to_adapt
        sigma = torch.zeros(orthogonal_size, orthogonal_size, device = device)
        sigma[:not_trainable_part_size, :not_trainable_part_size] = torch.eye(not_trainable_part_size, device = device)
        sigma[not_trainable_part_size:, not_trainable_part_size:] = torch.diag(diag_values)
        
        # if max_rank < left_size:
        #     sigma = torch.cat((sigma, torch.zeros(left_size - max_rank, max_rank, device = device)), dim=0)
        # elif max_rank < right_size:
        #     sigma = torch.cat((sigma, torch.zeros(max_rank, right_size - max_rank, device = device)), dim=1)
        
        return sigma
