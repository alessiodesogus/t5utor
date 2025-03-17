
import transformers
import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSeq2SeqLM
import os
import torch
from trl import DPOConfig, DPOTrainer

import datasets
from torch.utils.data import Dataset, DataLoader
import argparse
os.environ["WANDB_PROJECT"] = "MNLPredators"
cpu_workers = 4
def train(args):
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if "json" in args.dataset:
        data = load_dataset('json', data_files = args.dataset)
    else:
        data = load_dataset(args.dataset)
    start_size = len(data['train'])
    if args.filter:
        data = data.filter(lambda x: len(tokenizer(x['prompt'],truncation=False, padding=False)['input_ids']) <= args.max_prompt_length)
    
    print(f"Filtered {start_size - len(data['train'])} samples")
    data = data['train'].train_test_split(test_size = args.test_size, seed=args.seed)
    valid_data = data['test']
    train_data = data['train']
    columns_to_remove = [c for c in valid_data.column_names  if c not in ["prompt", "chosen", "rejected"]]
    valid_data = valid_data.remove_columns(columns_to_remove)
    train_data = train_data.remove_columns(columns_to_remove)

    folder = f"~/scratch/experiments/DPO/{os.path.basename(model_name)}-{os.path.basename(args.dataset)}-{args.num_epochs}epochs-{args.max_target_length}max_target_length-{args.batch_size}batch_size-{args.learning_rate}lr-{args.weight_decay}wd-{args.warmup_ratio}warmup-{args.gradient_accumulation_steps}grad_accum-{args.loss_type}loss_type/"

    training_args = DPOConfig(
        folder,
        beta=0.1,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        warmup_ratio=args.warmup_ratio,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        save_total_limit=args.save_total,
        num_train_epochs=args.num_epochs,
        load_best_model_at_end=True,
        bf16=False,
        report_to="wandb",
        run_name=f"DPO-{os.path.basename(model_name)}-{os.path.basename(args.dataset)}-{args.num_epochs}epochs-{args.max_target_length}max_target_length-{args.batch_size}batch_size-{args.learning_rate}lr-{args.weight_decay}wd-{args.warmup_ratio}warmup-{args.gradient_accumulation_steps}grad_accum-{args.loss_type}loss_type",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_prefetch_factor=cpu_workers,
        dataloader_num_workers=cpu_workers,
        max_prompt_length= args.max_prompt_length,
        max_target_length= args.max_target_length,
        max_length= args.max_prompt_length + args.max_target_length,
        precompute_ref_log_probs=True,
        dataset_num_proc=cpu_workers,
        auto_find_batch_size=True,
        save_safetensors=False,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=False,
    )
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        loss_type=args.loss_type,
        train_dataset=train_data,
        eval_dataset=valid_data,
        tokenizer=tokenizer,
    )

    dpo_trainer.train(resume_from_checkpoint=args.resume)
    os.mkdir(os.path.join(folder, "final"))
    dpo_trainer.save_model(os.path.join(folder, "final"))



if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Script for pretraining a language model")
    
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use for training")
    parser.add_argument("--model_name", type=str, required=True, help="Pretrained model name or path")

    parser.add_argument("--max_prompt_length", type=int, default=512, help="Maximum sequence length for prompt tokenization")
    parser.add_argument("--max_target_length", type=int, default=1024, help="Maximum sequence length for target tokenization")

    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--test_size", type=float, default=0.01, help="Test size for validation")
    parser.add_argument("--seed", type=int, default=42, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log training loss every X steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Initial warmup ratio for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for training")
    parser.add_argument("--save_steps", type=int, default=500, help="Save model checkpoint every X steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Gradient accumulation steps for training")
    parser.add_argument("--eval_steps", type=int, default=100, help="Save model checkpoint every X steps")
    parser.add_argument("--save_total", type=int, default=5, help="Total checkpoints to save")
    parser.add_argument("--filter", action="store_true", help="Filter samples that are too long")
    parser.add_argument("--loss_type", type=str, default="sigmoid", help="Loss type")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing")

    args = parser.parse_args()

    train(args)