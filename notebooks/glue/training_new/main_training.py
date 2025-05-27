import argparse
from training import train_model


# ---------------------------  main  --------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed",   type=int)
    parser.add_argument("--task",   type=str)
    parser.add_argument("--num_svalues_to_adapt", type=int)
    parser.add_argument("--num_svectors_to_adapt", type=int)
    parser.add_argument("--head_lr", type=float)
    parser.add_argument("--adapter_lr", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--max_len", type=int)
    parser.add_argument("--initial_scaler", type=float)
    parser.add_argument("--initial_sigma", type=float)
    parser.add_argument("--cuda_visible_devices", type=str)
    parser.add_argument("--base_model_id", type=str)
    parser.add_argument("--model_type", type=str, )
    parser.add_argument("--method_name", type=str)
    parser.add_argument("--uiortholora_alpha", type=float)
    parser.add_argument("--uiortholora_dropout", type=float)
    parser.add_argument("--uilinlora_alpha", type=float)
    parser.add_argument("--uilinlora_dropout", type=float)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--target_modules", nargs="+")

    args = parser.parse_args()


    train_model(args)


if __name__ == "__main__":
    main()
