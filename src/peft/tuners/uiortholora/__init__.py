from peft.tuners.uiortholora.config import UIOrthoLoRAConfig
from peft.tuners.uiortholora.model import UIOrthoLoRAModel
from peft.tuners.uiortholora.bnb import Linear8bitLt, Linear4bit
from peft.utils import register_peft_method

register_peft_method(name="uiortholora", config_cls=UIOrthoLoRAConfig, model_cls=UIOrthoLoRAModel)

__all__ = ["UIOrthoLoRAConfig", "UIOrthoLoRAModel", "Linear8bitLt", "Linear4bit"] 