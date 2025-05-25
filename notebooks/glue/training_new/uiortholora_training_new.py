import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch, evaluate, argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, BitsAndBytesConfig, Trainer
)
from peft import UIOrthoLoRAConfig, get_peft_model, TaskType
from datetime import datetime
from clearml import Task



# ---------------------------  custom trainer  --------------------------- #
class UIOrthoLoRATrainer(Trainer):
    def __init__(self, *args, head_lr=1e-3, adapter_lr=4e-3, **kw):
        super().__init__(*args, **kw)
        self.head_lr, self.adapter_lr = head_lr, adapter_lr

    def create_optimizer(self):                       # two learning rates
        if self.optimizer is None:
            head, adapter = [], []
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    (head if "classifier" in n else adapter).append(p)
            groups = [{"params": head,    "lr": self.head_lr},
                      {"params": adapter, "lr": self.adapter_lr}]
            self.optimizer = torch.optim.AdamW(groups)
        return self.optimizer

# ---------------------------  helpers  --------------------------- #
def prepare_dataset(tokenizer, max_len=128, task="sst2"):
    ds = load_dataset("glue", task)
    
    def tokenize_function(examples):
        if task in ["sst2", "cola"]:
            # Single sentence tasks
            return tokenizer(
                examples["sentence"],
                truncation=True,
                padding="max_length",
                max_length=max_len
            )
        elif task in ["mrpc", "qnli", "rte", "wnli", "mnli", "qqp", "sts-b"]:
            # Two sentence tasks
            return tokenizer(
                examples["sentence1"],
                examples["sentence2"],
                truncation=True,
                padding="max_length",
                max_length=max_len
            )
    
    ds = ds.map(tokenize_function, batched=True)
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return ds

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return eval_metrics.compute(predictions=preds, references=labels)

def get_eval_metric_type(task):
    if task == "cola":
        return "matthews_correlation"
    elif task in ["mrpc", "qqp"]:
        return "f1"
    elif task in ["qnli", "rte", "wnli", "sst2", "mnli"]:
        return "accuracy"
    elif task == "sts-b":
        return "pearson"
    else:
        raise ValueError(f"Unsupported task: {task}")


def print_trainable_params(model):
    total = 0
    trainable = 0
    for name, param in model.named_parameters():
        if "classifier" in name:
            continue
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    print(f"Trainable parameters (excluding classifier): {trainable:,}")
    print(f"Total parameters (excluding classifier): {total:,}")
    print(f"Trainable %: {100 * trainable / total:.8f}%")


def write_results(score, timestamp, args):
    output_path = f"results_{args.task}_{timestamp}.txt"

    if not os.path.exists(output_path):
        with open(output_path, "w") as f:
            f.write("rank,head_lr,adapter_lr,scaler,sigma,score\n")

    with open(output_path, "a") as f:
        # f.write(f"{args.rank},{args.head_lr},{args.adapter_lr},{args.initial_scaler},{args.initial_sigma},{score:.4f}\n")
        f.write(f"{args.num_svalues_to_adapt},{args.num_svectors_to_adapt},{args.head_lr},{args.adapter_lr},{args.initial_scaler},{args.initial_sigma},{score:.8f}\n")




# ---------------------------  main  --------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--task",   type=str, default="sst2")
    parser.add_argument("--num_svalues_to_adapt",   type=int, default=128)
    parser.add_argument("--num_svectors_to_adapt",   type=int, default=20)
    parser.add_argument("--head_lr",   type=float, default=4e-3)
    parser.add_argument("--adapter_lr",   type=float, default=4e-3)
    parser.add_argument("--batch_size",   type=int, default=64)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--initial_scaler", type=float, default=1.0)
    parser.add_argument("--initial_sigma", type=float, default=1.0)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    task = Task.init(
        project_name="GLUE benchmark",
        task_name=f"UIOrthoLoRA tuner - {args.task} - {timestamp}"
    )
    task.connect(args)

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.set_printoptions(threshold=float("inf"))
    eval_metric_type = get_eval_metric_type(args.task)
    global eval_metrics; eval_metrics = evaluate.load(eval_metric_type)

    base_id = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(base_id, use_fast=True)

    base = AutoModelForSequenceClassification.from_pretrained(
        base_id, num_labels=2, device_map="auto"
    )

    uiortholora_cfg = UIOrthoLoRAConfig(
            target_modules=["query", "value"],
            uiortholora_alpha=1.0,
            uiortholora_dropout=0.0,
            fan_in_fan_out=False,
            initial_scaler=args.initial_scaler,
            initial_sigma=args.initial_sigma,
            num_svalues_to_adapt=args.num_svalues_to_adapt,
            num_svectors_to_adapt=args.num_svectors_to_adapt,
            task_type=TaskType.SEQ_CLS)
    model = get_peft_model(base, uiortholora_cfg)

    for m in model.modules():
        if hasattr(m, "parametrizations") and "weight" in m.parametrizations:
            base = m.parametrizations.weight[0].base
            if not base.is_contiguous():
                base.data = base.data.contiguous()


    model.classifier.requires_grad_(True)   # make head trainable
    model.config.pad_token_id = tokenizer.pad_token_id
    print_trainable_params(model)

    data = prepare_dataset(tokenizer, task=args.task, max_len=args.max_len)

    train_args = TrainingArguments(
        output_dir=f"uiortholora-roberta-base-{args.task}",
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=eval_metric_type,
        greater_is_better=True,
        warmup_ratio=0.06,
        lr_scheduler_type="linear",
        logging_steps=50,
        save_total_limit=1,
        seed=args.seed,
    )

    os.makedirs(f"uiortholora-roberta-base-{args.task}", exist_ok=True)

    trainer = UIOrthoLoRATrainer(
        model=model,
        args=train_args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        head_lr=args.head_lr,
        adapter_lr=args.adapter_lr
    )

    trainer.train()

    score = trainer.evaluate()[f"eval_{eval_metric_type}"]
    print(f"Final {eval_metric_type}:", score)
    write_results(score, timestamp, args)


if __name__ == "__main__":
    main()
