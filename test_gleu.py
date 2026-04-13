import os

import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, DebertaV2Tokenizer, T5Tokenizer, get_scheduler

from Debertav2_transformers import DebertaV2Config, DebertaV2ForSequenceClassification
from T5_transformers import T5Config, T5ForSequenceClassification
from utils import (
    build_metric_tracking_metadata,
    maybe_skip_completed_run,
    maybe_dispatch_multi_seed_runs,
    parse_args,
    save_run_metrics,
    seed_fix,
    tf_make_result_path,
    update_best_metrics,
)

MODEL_LIST = {
    "deberta": {
        "tokenizer": DebertaV2Tokenizer,
        "model": DebertaV2ForSequenceClassification,
        "config": DebertaV2Config,
        "model_load_path": "microsoft/deberta-v3-large",
    },
    "t5": {
        "tokenizer": T5Tokenizer,
        "model": T5ForSequenceClassification,
        "config": T5Config,
        "model_load_path": "google-t5/t5-base",
    },
}

TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
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

args.result_path = tf_make_result_path(args)
if args.dev:
    args.result_path = "test_" + args.result_path

task = args.glue_task
model_utils = MODEL_LIST[model_type]
if args.model_load_path is not None:
    model_utils["model_load_path"] = args.model_load_path

if maybe_skip_completed_run(args, model_name=model_type, task_name=task):
    raise SystemExit(0)

os.makedirs(args.data_path, exist_ok=True)
dataset = load_dataset("glue", task, cache_dir=args.data_path)
print(dataset)

num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
tokenizer = model_utils["tokenizer"].from_pretrained(model_utils["model_load_path"])


def build_metric():
    return load_metric("glue", task, cache_dir=args.data_path, trust_remote_code=True)


def custom_collate_fn(batches):
    sentence1_key, sentence2_key = TASK_TO_KEYS[task]
    if sentence2_key is None:
        texts = [batch[sentence1_key] for batch in batches]
    else:
        texts = [batch[sentence1_key] + "\n" + batch[sentence2_key] for batch in batches]

    tokenized_inputs = tokenizer(
        texts,
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
        padding=True,
    )

    labels = [batch["label"] for batch in batches]
    tokenized_inputs["labels"] = torch.LongTensor(labels)
    if task == "stsb":
        tokenized_inputs["labels"] = torch.FloatTensor(labels)
    return tokenized_inputs


train_dataloader = DataLoader(
    dataset["train"],
    batch_size=args.batch_size,
    collate_fn=custom_collate_fn,
    num_workers=4,
    shuffle=True,
)

if task == "mnli":
    val_dataloader = DataLoader(
        dataset["validation_matched"],
        batch_size=args.batch_size,
        collate_fn=custom_collate_fn,
        num_workers=4,
    )
    test_dataloader = DataLoader(
        dataset["validation_mismatched"],
        batch_size=args.batch_size,
        collate_fn=custom_collate_fn,
        num_workers=4,
    )
else:
    try:
        validation_split = dataset["validation"]
    except KeyError:
        validation_split = dataset["test"]
    val_dataloader = DataLoader(
        validation_split,
        batch_size=args.batch_size,
        collate_fn=custom_collate_fn,
        num_workers=4,
    )
    test_dataloader = DataLoader(
        dataset["test"],
        batch_size=args.batch_size,
        collate_fn=custom_collate_fn,
        num_workers=4,
    )

if "wnli" in args.glue_task or "qqp" in args.glue_task:
    test_dataloader = DataLoader(
        dataset["validation"],
        batch_size=args.batch_size,
        collate_fn=custom_collate_fn,
        num_workers=4,
    )

model = model_utils["model"].from_pretrained(model_utils["model_load_path"], num_labels=num_labels)

if model_type == "deberta":
    pre_trained_model = model.deberta
else:
    model.config.problem_type = "single_label_classification"
    pre_trained_model = model.transformer

if task == "stsb":
    model.config.problem_type = "regression"

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
        pre_trained_model.add_unit_init_before_dotpro(
            layer_num=args.add_linear_layer,
            head_indi=args.head_indi,
            init_type=args.init_type,
            act_type=args.act_type,
        )
    elif args.add_position == "afterffnn":
        pre_trained_model.add_unit_init_after_ffnn(
            layer_num=args.add_linear_layer,
            init_type=args.init_type,
            act_type=args.act_type,
        )
    elif args.add_position == "both":
        pre_trained_model.add_unit_init_after_ffnn(
            layer_num=args.add_linear_layer,
            init_type=args.init_type,
            act_type=args.act_type,
        )
        pre_trained_model.add_unit_init_before_dotpro(
            layer_num=args.add_linear_layer,
            head_indi=args.head_indi,
            init_type=args.init_type,
            act_type=args.act_type,
        )
    elif args.add_position == "aftffnn1":
        pre_trained_model.add_unit_init_after_ffnn1(
            layer_num=args.add_linear_layer,
            init_type=args.init_type,
            act_type=args.act_type,
        )
    elif args.add_position == "aftffnn2":
        pre_trained_model.add_unit_init_after_ffnn2(
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


def maybe_collect_relative_perturbation():
    if args.diagnostic_mode != "relative_perturbation":
        return {}
    if model_type != "deberta" or args.no_add_linear:
        return {}

    hook_values = []
    handles = []

    def hook_fn(module, module_inputs, module_output):
        if not module_inputs:
            return
        input_tensor = module_inputs[0]
        output_tensor = module_output
        if not torch.is_tensor(input_tensor) or not torch.is_tensor(output_tensor):
            return

        input_tensor = input_tensor.detach()
        output_tensor = output_tensor.detach()
        denominator = torch.linalg.vector_norm(input_tensor)
        if denominator.item() == 0:
            return
        numerator = torch.linalg.vector_norm(output_tensor - input_tensor)
        hook_values.append((numerator / denominator).item())

    for module_name, module in model.named_modules():
        if ("added_layer" in module_name or "added_layers" in module_name) and hasattr(module, "weight"):
            handles.append(module.register_forward_hook(hook_fn))

    if not handles:
        return {}

    was_training = model.training
    model.eval()
    with torch.no_grad():
        first_batches = next(iter(train_dataloader))
        for key in first_batches.keys():
            first_batches[key] = first_batches[key].to(device)
        model(**first_batches)

    for handle in handles:
        handle.remove()
    if was_training:
        model.train()

    if not hook_values:
        return {}

    return {
        "relative_perturbation_mean": sum(hook_values) / len(hook_values),
        "relative_perturbation_count": len(hook_values),
    }


model.to(device)
print(model)

args.model_config = model.config
save_path = None
if not args.skip_checkpoint_save:
    save_path = "checkpoints/" + args.result_path + "_" + task
    os.makedirs(save_path, exist_ok=True)

optimizer = AdamW(
    model.parameters(),
    lr=args.learning_rate,
    betas=[args.beta1, args.beta2],
    weight_decay=args.weight_decay,
    eps=args.eps,
)
scheduler = get_scheduler("linear", optimizer, args.warmup_steps, len(train_dataloader) * args.epoch)

for name, param in model.named_parameters():
    if "added" in name:
        print(name, param)
        break

diagnostics = maybe_collect_relative_perturbation()
latest_metrics = {}
score_history = []
best_metrics = None
best_metric_name = None
best_metric_value = None
best_metric_maximize = None

for epoch_idx in range(1, args.epoch + 1):
    model.train()
    train_losses = []
    train_loop = tqdm(train_dataloader)
    for batches in train_loop:
        for key in batches.keys():
            batches[key] = batches[key].to(device)

        out = model(**batches)
        out.loss.backward()
        train_losses.append(out.loss.item())

        if not args.adapter:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        train_loop.set_description("loss=" + str(out.loss.item()))

    train_loss = sum(train_losses) / len(train_losses)
    print("train_loss = {}".format(train_loss))

    eval_metric = build_metric()
    model.eval()
    eval_losses = []
    with torch.no_grad():
        for batches in tqdm(val_dataloader):
            for key in batches.keys():
                batches[key] = batches[key].to(device)

            out = model(**batches)
            eval_losses.append(out.loss.item())
            if num_labels == 1:
                eval_metric.add_batch(predictions=out.logits, references=batches["labels"])
            else:
                eval_metric.add_batch(predictions=torch.argmax(out.logits, dim=-1), references=batches["labels"])

    final_score = eval_metric.compute()
    dev_loss = sum(eval_losses) / len(eval_losses)
    print("dev_loss = {}".format(dev_loss))
    print("dev", final_score)

    latest_metrics = {
        "{}_train_loss".format(task): train_loss,
        "{}_dev_loss".format(task): dev_loss,
        "epoch": epoch_idx,
    }
    for metric_name, metric_value in final_score.items():
        latest_metrics["{}_dev_{}".format(task, metric_name)] = metric_value
    score_history.append(dict(latest_metrics))
    best_metrics, best_metric_name, best_metric_value, best_metric_maximize = update_best_metrics(
        best_metrics,
        best_metric_name,
        best_metric_value,
        best_metric_maximize,
        latest_metrics,
    )

    for name, param in model.named_parameters():
        if "added" in name:
            print(name, param)
            break

    if save_path is not None:
        model_name = "model_" + str(epoch_idx) + ".pt"
        torch.save(model.state_dict(), os.path.join(save_path, model_name))

metrics_path = save_run_metrics(
    args,
    model_name=model_type,
    task_name=task,
    final_metrics=latest_metrics,
    checkpoint_dir=save_path,
    extra_metadata={
        "script": "test_gleu.py",
        "diagnostics": diagnostics,
        **build_metric_tracking_metadata(
            score_history=score_history,
            best_metrics=best_metrics,
            best_metric_name=best_metric_name,
            best_metric_value=best_metric_value,
        ),
    },
)
print("saved metrics to", metrics_path)
