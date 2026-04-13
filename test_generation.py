import os

import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, DataCollatorForSeq2Seq, T5Tokenizer

from T5_transformers import T5Config, T5ForConditionalGeneration
from utils import (
    build_metric_tracking_metadata,
    gen_make_result_path,
    maybe_skip_completed_run,
    maybe_dispatch_multi_seed_runs,
    parse_args,
    save_run_metrics,
    seed_fix,
    update_best_metrics,
)

MODEL_LIST = {
    "t5": {
        "tokenizer": T5Tokenizer,
        "model": T5ForConditionalGeneration,
        "config": T5Config,
        "model_load_path": "google-t5/t5-base",
    }
}


args = parse_args()
if maybe_dispatch_multi_seed_runs(args):
    raise SystemExit(0)

seed_fix(args.seed)
device = "cuda:" + str(args.gpu)

model_type = None
for candidate_model_type in MODEL_LIST:
    if candidate_model_type in args.result_path.split("_"):
        model_type = candidate_model_type
        break
if model_type is None:
    raise ValueError("result_path must include model type")

args.result_path = gen_make_result_path(args)
if args.dev:
    args.result_path = "test_" + args.result_path

task = args.generation_task
model_utils = MODEL_LIST[model_type]
if args.model_load_path is not None:
    model_utils["model_load_path"] = args.model_load_path

if maybe_skip_completed_run(args, model_name=model_type, task_name=task):
    raise SystemExit(0)

os.makedirs(args.data_path, exist_ok=True)
if task == "cnndm":
    dataset = load_dataset("cnn_dailymail", "3.0.0", cache_dir=args.data_path)
else:
    dataset = load_dataset("wmt16", "ro-en", cache_dir=args.data_path)
print(dataset)

tokenizer = model_utils["tokenizer"].from_pretrained(model_utils["model_load_path"])
pad_token_id = tokenizer.pad_token_id
print(pad_token_id)


def build_metric():
    if task == "cnndm":
        return load_metric("rouge", cache_dir=args.data_path)
    return load_metric("sacrebleu", cache_dir=args.data_path, trust_remote_code=True)


model = model_utils["model"].from_pretrained(model_utils["model_load_path"])


def preprocess_function(examples):
    if "wmt" in task:
        inputs = [example["en"] for example in examples["translation"]]
        targets = [example["ro"] for example in examples["translation"]]
    else:
        inputs = examples["article"]
        targets = examples["highlights"]

    model_inputs = tokenizer(inputs, max_length=tokenizer.model_max_length, padding=False, truncation=True)
    labels = tokenizer(text_target=targets, max_length=tokenizer.model_max_length, padding=False, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


column_names = dataset["train"].column_names
dataset["train"] = dataset["train"].map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=column_names,
)
dataset["validation"] = dataset["validation"].map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=column_names,
)
dataset["test"] = dataset["test"].map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=column_names,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100,
)

print(dataset["train"])

train_dataloader = DataLoader(
    dataset["train"],
    batch_size=args.batch_size,
    collate_fn=data_collator,
    num_workers=4,
    shuffle=True,
)
val_dataloader = DataLoader(
    dataset["validation"],
    batch_size=args.batch_size,
    collate_fn=data_collator,
    num_workers=4,
)
test_dataloader = DataLoader(
    dataset["test"],
    batch_size=args.batch_size,
    collate_fn=data_collator,
    num_workers=4,
)

if not args.no_add_linear:
    if args.add_linear_layer is None:
        if args.add_linear_num is None:
            args.add_linear_layer = range(model.config.num_hidden_layers)
        elif args.add_linear_num > 0:
            args.add_linear_layer = range(args.add_linear_num)
        else:
            args.add_linear_layer = range(
                model.config.num_hidden_layers + args.add_linear_num,
                model.config.num_hidden_layers,
            )

    if args.add_position == "befdot":
        model.add_unit_init_before_dotpro(
            layer_num=args.add_linear_layer,
            head_indi=args.head_indi,
            init_type=args.init_type,
            act_type=args.act_type,
        )
    elif args.add_position == "afterffnn":
        model.add_unit_init_after_ffnn(
            layer_num=args.add_linear_layer,
            init_type=args.init_type,
            act_type=args.act_type,
        )
    elif args.add_position == "both":
        model.add_unit_init_after_ffnn(
            layer_num=args.add_linear_layer,
            init_type=args.init_type,
            act_type=args.act_type,
        )
        model.add_unit_init_before_dotpro(
            layer_num=args.add_linear_layer,
            head_indi=args.head_indi,
            init_type=args.init_type,
            act_type=args.act_type,
        )
    elif args.add_position == "aftffnn1":
        model.add_unit_init_after_ffnn1(
            layer_num=args.add_linear_layer,
            init_type=args.init_type,
            act_type=args.act_type,
        )
    elif args.add_position == "aftffnn2":
        model.add_unit_init_after_ffnn2(
            layer_num=args.add_linear_layer,
            init_type=args.init_type,
            act_type=args.act_type,
        )

if args.adapter:
    for name, param in model.named_parameters():
        if "added" not in name and "classifi" not in name:
            param.requires_grad_(requires_grad=False)
        else:
            param.requires_grad_(requires_grad=True)

    for name, param in model.named_parameters():
        print(param.requires_grad, "\t/\t", name)

model.to(device)
print(model)
print(args)

args.model_config = model.config

optimizer = AdamW(
    model.parameters(),
    lr=args.learning_rate,
    betas=[args.beta1, args.beta2],
    weight_decay=args.weight_decay,
    eps=args.eps,
)

for name, param in model.named_parameters():
    if "added" in name:
        print(name, param)
        break


def evaluate(current_steps, current_train_loss, current_update_steps):
    for name, param in model.named_parameters():
        if "added" in name:
            print(name, param)
            break

    metric = build_metric()
    model.eval()
    with torch.no_grad():
        for batches in tqdm(test_dataloader):
            for key in batches.keys():
                batches[key] = batches[key].to(device)

            out = model.generate(
                input_ids=batches["input_ids"],
                attention_mask=batches["attention_mask"],
                num_beams=4,
                max_new_tokens=300,
            )
            decode_pred = tokenizer.batch_decode(out, skip_special_tokens=True)

            labels = batches["labels"].clone()
            labels.masked_fill_(labels == -100, pad_token_id)
            labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            decode_pred = [item.strip() for item in decode_pred]
            if task == "cnndm":
                labels = [item.strip() for item in labels]
            else:
                labels = [[item.strip()] for item in labels]

            metric.add_batch(predictions=decode_pred, references=labels)

    final_score = metric.compute()
    print("test")
    print(final_score)

    metrics = {
        "{}_train_loss".format(task): current_train_loss,
        "steps": current_steps,
        "update_steps": current_update_steps,
    }
    if task == "cnndm":
        metrics["cnndm_test_rouge1"] = final_score["rouge1"].mid.fmeasure
        metrics["cnndm_test_rouge2"] = final_score["rouge2"].mid.fmeasure
        metrics["cnndm_test_rougeL"] = final_score["rougeL"].mid.fmeasure
        metrics["cnndm_test_rougeLsum"] = final_score["rougeLsum"].mid.fmeasure
    else:
        metrics["wmt_en_ro_bleu"] = final_score["score"]
        metrics["wmt_en_ro_precision1"] = final_score["precisions"][0]
        metrics["wmt_en_ro_precision2"] = final_score["precisions"][1]
        metrics["wmt_en_ro_precision3"] = final_score["precisions"][2]
        metrics["wmt_en_ro_precision4"] = final_score["precisions"][3]

    model.train()
    return metrics


steps = 1
update_step = 1
last_train_loss = None
latest_metrics = {}
score_history = []
best_metrics = None
best_metric_name = None
best_metric_value = None
best_metric_maximize = None
stop_training = False
accumulation_steps = max(1, args.generate_full_batch // args.batch_size)

for epoch_idx in range(1, args.epoch + 1):
    model.train()
    losses = []
    train_loop = tqdm(train_dataloader)
    for batches in train_loop:
        for key in batches.keys():
            batches[key] = batches[key].to(device)

        out = model(**batches)
        out.loss.backward()
        losses.append(out.loss.item())

        if steps % accumulation_steps == 0 or steps == len(train_dataloader) - 1:
            optimizer.step()
            optimizer.zero_grad()
            update_step += 1

        train_loop.set_description("loss=" + str(out.loss.item()))

        if steps % args.logging_step == 0:
            current_train_loss = sum(losses) / len(losses)
            latest_metrics = evaluate(steps, current_train_loss, update_step - 1)
            score_history.append(dict(latest_metrics))
            best_metrics, best_metric_name, best_metric_value, best_metric_maximize = update_best_metrics(
                best_metrics,
                best_metric_name,
                best_metric_value,
                best_metric_maximize,
                latest_metrics,
            )

        if update_step > args.gen_train_step:
            stop_training = True
            break

        steps += 1

    if losses:
        last_train_loss = sum(losses) / len(losses)
        print("train_loss = {}".format(last_train_loss))

    if stop_training:
        break

if last_train_loss is None:
    last_train_loss = 0.0

final_step = max(steps - 1, 1)
if not latest_metrics or latest_metrics.get("steps") != final_step:
    latest_metrics = evaluate(final_step, last_train_loss, update_step - 1)
    score_history.append(dict(latest_metrics))
    best_metrics, best_metric_name, best_metric_value, best_metric_maximize = update_best_metrics(
        best_metrics,
        best_metric_name,
        best_metric_value,
        best_metric_maximize,
        latest_metrics,
    )

metrics_path = save_run_metrics(
    args,
    model_name=model_type,
    task_name=task,
    final_metrics=latest_metrics,
    checkpoint_dir=None,
    extra_metadata={
        "script": "test_generation.py",
        "evaluation_split": "test",
        **build_metric_tracking_metadata(
            score_history=score_history,
            best_metrics=best_metrics,
            best_metric_name=best_metric_name,
            best_metric_value=best_metric_value,
        ),
    },
)
print("saved metrics to", metrics_path)
