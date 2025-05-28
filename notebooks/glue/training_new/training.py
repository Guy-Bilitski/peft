import os
import torch, evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from peft import UIOrthoLoRAConfig, UILinLoRAConfig, get_peft_model, TaskType
from datetime import datetime
from clearml import Task
import pandas as pd



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


def write_results(score, timestamp, args, base_dir="results/glue"):
    os.makedirs(base_dir, exist_ok=True)
    csv_path = os.path.join(base_dir, f"{args.model_type.lower()}_{args.task}.csv")

    row = {
        "num_svalues": args.num_svalues_to_adapt,
        "num_svectors": args.num_svectors_to_adapt,
        "head_lr": args.head_lr,
        "adapter_lr": args.adapter_lr,
        "scaler": args.initial_scaler,
        "sigma": args.initial_sigma,
        "score": round(score, 8),
        "timestamp": timestamp
    }

    df = pd.DataFrame([row])

    # Append or create the CSV
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', index=False, header=False)
    else:
        df.to_csv(csv_path, mode='w', index=False, header=True)


def is_duplicate_run(args, base_dir="results/glue"):
    if getattr(args, "resume_from_checkpoint", None):
        return False
    csv_path = os.path.join(base_dir, f"{args.model_type.lower()}_{args.task}.csv")
    if not os.path.exists(csv_path):
        return False  # No file yet

    try:
        df_existing = pd.read_csv(csv_path)
        duplicate = df_existing[
            (df_existing["num_svalues"] == args.num_svalues_to_adapt) &
            (df_existing["num_svectors"] == args.num_svectors_to_adapt) &
            (df_existing["head_lr"] == args.head_lr) &
            (df_existing["adapter_lr"] == args.adapter_lr) &
            (df_existing["scaler"] == args.initial_scaler) &
            (df_existing["sigma"] == args.initial_sigma)
        ]
        return not duplicate.empty
    except Exception as e:
        print(f"Warning: couldn't check for duplicates — proceeding anyway. Reason: {e}")
        return False




def get_peft_config(args):
    print(args)
    if args.model_type == "uiortholora":
        return UIOrthoLoRAConfig(
                target_modules=args.target_modules,
                uiortholora_alpha=args.uiortholora_alpha,
                uiortholora_dropout=args.uiortholora_dropout,
                fan_in_fan_out=False,
                initial_scaler=args.initial_scaler,
                initial_sigma=args.initial_sigma,
                num_svalues_to_adapt=args.num_svalues_to_adapt,
                num_svectors_to_adapt=args.num_svectors_to_adapt,
                task_type=TaskType.SEQ_CLS)
    elif args.model_type == "uilinlora":
        return UILinLoRAConfig(
                target_modules=args.target_modules,
                uilinlora_alpha=args.uilinlora_alpha,
                uilinlora_dropout=args.uilinlora_dropout,
                fan_in_fan_out=False,
                initial_scaler=args.initial_scaler,
                initial_sigma=args.initial_sigma,
                rank=args.rank,
                task_type=TaskType.SEQ_CLS)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
def set_contiguous(model):
    for m in model.modules():
        if hasattr(m, "parametrizations") and "weight" in m.parametrizations:
            base = m.parametrizations.weight[0].base
            if not base.is_contiguous():
                base.data = base.data.contiguous()

def prepare_trainer(model, args, data, tokenizer, eval_metric_type, timestamp):
    train_args = TrainingArguments(
        output_dir = f"outputs/{args.model_type.lower()}_{args.base_model_id.replace('/', '-')}_{args.task}_{timestamp}",
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

    return trainer


def train_model(args):
    if is_duplicate_run(args):
        print("Duplicate run detected. Skipping...")
        return

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    task = Task.init(
        project_name="GLUE benchmark",
        task_name=f"UIOrthoLoRA tuner - {args.task} - {timestamp}"
    )
    task.connect(args)

    torch.set_printoptions(threshold=float("inf"))
    eval_metric_type = get_eval_metric_type(args.task)
    global eval_metrics; eval_metrics = evaluate.load(eval_metric_type)

    base_model_id = args.base_model_id
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_id, num_labels=2, device_map="auto"
    )

    peft_config = get_peft_config(args)
    model = get_peft_model(base_model, peft_config)

    if args.model_type == "uiortholora":
        set_contiguous(model)

    model.classifier.requires_grad_(True)   # make head trainable
    model.config.pad_token_id = tokenizer.pad_token_id

    print("Trainable parameters with head:")
    model.print_trainable_parameters()

    print("Trainable parameters without head:")
    print_trainable_params(model)

    data = prepare_dataset(tokenizer, task=args.task, max_len=args.max_len)

    trainer = prepare_trainer(model, args, data, tokenizer, eval_metric_type, timestamp)

    if getattr(args, "resume_from_checkpoint", None):
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()


    score = trainer.evaluate()[f"eval_{eval_metric_type}"]
    print(f"Final {eval_metric_type}:", score)
    write_results(score, timestamp, args)
    task.close()

