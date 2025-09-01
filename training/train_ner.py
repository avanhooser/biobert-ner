
# BioBERT NER fine-tuning (CoNLL-style) with optional PEFT/LoRA adapters.
# Usage (CPU okay for tiny samples; use GPU on Colab/Studio Lab for real runs):
#   python training/train_ner.py --model dmis-lab/biobert-v1.1 --train data/train.conll --valid data/dev.conll --use_lora
import os, argparse, numpy as np
from typing import List, Dict, Tuple
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification,
                          TrainingArguments, Trainer)
import evaluate
import sys

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

def read_conll(path: str) -> Tuple[List[List[str]], List[List[str]]]:
    tokens, tags, cur_t, cur_y = [], [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if cur_t:
                    tokens.append(cur_t); tags.append(cur_y)
                    cur_t, cur_y = [], []
                continue
            # token [tab or space] tag
            parts = line.split()
            if len(parts) >= 2:
                cur_t.append(parts[0])
                cur_y.append(parts[-1])
    if cur_t:
        tokens.append(cur_t); tags.append(cur_y)
    return tokens, tags

def build_dataset(tokens: List[List[str]], tags: List[List[str]]):
    return Dataset.from_dict({"tokens": tokens, "ner_tags": tags})

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    prev_word_id = None
    for word_id in word_ids:
        if word_id is None:
            new_labels.append(-100)
        elif word_id != prev_word_id:
            new_labels.append(label2id[labels[word_id]])
        else:
            # Inside subword -> assign -100 or I-*; we keep -100 for simplicity
            new_labels.append(-100)
        prev_word_id = word_id
    return new_labels

def compute_metrics(p):
    # Correctly unpack predictions and labels from EvalPrediction object
    preds, refs = p.predictions, p.label_ids
    preds = np.argmax(preds, axis=-1)
    true_preds, true_labels = [], []
    # Modify to iterate only over preds and labels, relying on -100 for alignment
    for pred, label in zip(preds, p.label_ids):
        # align
        pred_tags, label_tags = [], []
        for p_i, l_i in zip(pred, label):
            if l_i != -100: # Use -100 label to filter tokens
                pred_tags.append(id2label[p_i])
                label_tags.append(id2label[l_i])
        true_preds.append(pred_tags); true_labels.append(label_tags)

    metric = evaluate.load("seqeval")
    results = metric.compute(predictions=true_preds, references=true_labels)
    return {"precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dmis-lab/biobert-v1.1")
    ap.add_argument("--train", default="training/data/sample.conll")
    ap.add_argument("--valid", default="training/data/sample.conll")
    ap.add_argument("--out", default="training/output")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--use_lora", action="store_true")

    # Parse known arguments and ignore unknown ones
    args, unknown = ap.parse_known_args()
    # If running in Colab notebook, unknown arguments like -f will be passed
    # and we can safely ignore them.

    train_tokens, train_tags = read_conll(args.train)
    valid_tokens, valid_tags = read_conll(args.valid)

    # derive labels from training set
    unique_labels = sorted({t for seq in train_tags for t in seq})
    global label2id, id2label
    label2id = {l:i for i,l in enumerate(unique_labels)}
    id2label = {i:l for l,i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    def tok(examples):
        tokenized = tokenizer(examples["tokens"], is_split_into_words=True, truncation=True, padding=False)
        labels = []
        for i, word_ids in enumerate([tokenized.word_ids(k) for k in range(len(examples["tokens"]))]):
            labels.append(align_labels_with_tokens(examples["ner_tags"][i], word_ids))
        tokenized["labels"] = labels
        return tokenized

    ds_train = build_dataset(train_tokens, train_tags).map(tok, batched=True)
    ds_valid = build_dataset(valid_tokens, valid_tags).map(tok, batched=True)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=len(unique_labels), id2label=id2label, label2id=label2id
    )

    if args.use_lora:
        if not PEFT_AVAILABLE:
            raise RuntimeError("peft not installed but --use_lora was provided.")
        lora = LoraConfig(r=8, lora_alpha=16, target_modules=["query","value","key","dense"], lora_dropout=0.05, bias="none")
        model = get_peft_model(model, lora)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    os.makedirs(args.out, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=args.out,
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch", # Changed from evaluation_strategy
        save_strategy="epoch",
        logging_steps=50,
        report_to=[],
        include_inputs_for_metrics=True, # Added to include attention_mask in compute_metrics
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_valid,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print("Saved model to", args.out)
