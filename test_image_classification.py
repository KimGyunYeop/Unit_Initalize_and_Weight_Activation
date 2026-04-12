import os

import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, ConvNextForImageClassification, ViTImageProcessor
from transformers import get_cosine_schedule_with_warmup

from modeling_vit import ViTConfig, ViTForImageClassification
from utils import (
    maybe_skip_completed_run,
    maybe_dispatch_multi_seed_runs,
    parse_args,
    save_run_metrics,
    seed_fix,
    tf_make_result_path,
)

MODEL_LIST = {
    "vit": {
        "model": ViTForImageClassification,
        "image_processor": ViTImageProcessor,
        "model_load_path": "google/vit-base-patch16-224-in21k",
        "image_processor_load_path": "google/vit-base-patch16-384",
    },
    "convnext": {
        "model": ConvNextForImageClassification,
        "image_processor": AutoImageProcessor,
        "model_load_path": "facebook/convnext-tiny-224",
    },
}

DATASET_TO_IMAGEDICT = {
    "cifar10": "img",
    "cifar100": "img",
    "imagenet-1k": "image",
}
DATASET_TO_LABELDICT = {
    "cifar10": "label",
    "cifar100": "fine_label",
    "imagenet-1k": "label",
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

model_utils = MODEL_LIST[model_type]
if args.model_load_path is not None:
    model_utils["model_load_path"] = args.model_load_path

dataset_name = args.image_classification_dataset
dataset_labeldict = DATASET_TO_LABELDICT[dataset_name]
dataset_imagedict = DATASET_TO_IMAGEDICT[dataset_name]

if maybe_skip_completed_run(args, model_name=model_type, task_name=dataset_name):
    raise SystemExit(0)

os.makedirs(args.data_path, exist_ok=True)
dataset = load_dataset(dataset_name, cache_dir=args.data_path, use_auth_token=True)

if dataset_name in ["cifar10", "cifar100"]:
    eval_step = args.cifar_eval_step
else:
    eval_step = args.imagenet_eval_step

processor = model_utils["image_processor"].from_pretrained(
    model_utils.get("image_processor_load_path", model_utils["model_load_path"])
)


def build_metric():
    return load_metric("accuracy")


def custom_collate_fn(batches):
    if dataset_name == "imagenet-1k":
        inputs = processor([batch[dataset_imagedict].convert("RGB") for batch in batches], return_tensors="pt")
    else:
        inputs = processor([batch[dataset_imagedict] for batch in batches], return_tensors="pt")

    inputs["labels"] = torch.tensor([batch[dataset_labeldict] for batch in batches])
    return inputs


train_dataloader = DataLoader(
    dataset["train"],
    batch_size=args.batch_size,
    collate_fn=custom_collate_fn,
    num_workers=4,
    shuffle=True,
)
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
if dataset_name == "imagenet-1k":
    test_dataloader = DataLoader(
        dataset["validation"],
        batch_size=args.batch_size,
        collate_fn=custom_collate_fn,
        num_workers=4,
    )

labels = dataset["train"].features[dataset_labeldict].names

model = model_utils["model"].from_pretrained(
    model_utils["model_load_path"],
    num_labels=len(labels),
    ignore_mismatched_sizes=True,
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
    print(args.add_linear_layer)
    if args.add_position == "befdot":
        model.vit.add_unit_init_before_dotpro(
            layer_num=args.add_linear_layer,
            head_indi=args.head_indi,
            init_type=args.init_type,
            act_type=args.act_type,
        )
    elif args.add_position == "afterffnn":
        model.vit.add_unit_init_after_ffnn(
            layer_num=args.add_linear_layer,
            init_type=args.init_type,
            act_type=args.act_type,
        )
    elif args.add_position == "both":
        model.vit.add_unit_init_after_ffnn(
            layer_num=args.add_linear_layer,
            init_type=args.init_type,
            act_type=args.act_type,
        )
        model.vit.add_unit_init_before_dotpro(
            layer_num=args.add_linear_layer,
            head_indi=args.head_indi,
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

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=args.learning_rate,
    momentum=args.momentum,
    weight_decay=args.weight_decay,
)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=args.warmup_steps,
    num_training_steps=args.epoch * len(train_dataloader),
)

args.model_config = model.config
save_path = None
if not args.skip_checkpoint_save:
    save_path = "checkpoints/" + args.result_path + "_" + args.image_classification_dataset
    os.makedirs(save_path, exist_ok=True)

print("Model Type:", model_type)
print("Dataset Name:", dataset_name)


def evaluate(loader, split_name):
    metric = build_metric()
    losses = []
    model.eval()
    with torch.no_grad():
        for batches in tqdm(loader):
            for key in batches.keys():
                batches[key] = batches[key].to(device)
            batches["interpolate_pos_encoding"] = True

            out = model(**batches)
            losses.append(out.loss.item())
            metric.add_batch(predictions=torch.argmax(out.logits, dim=-1), references=batches["labels"])

    final_score = metric.compute()
    average_loss = sum(losses) / len(losses)
    print("{}_loss = {}".format(split_name, average_loss))
    print(split_name, final_score)

    metrics = {
        "{}_{}_loss".format(dataset_name, split_name): average_loss,
    }
    for metric_name, metric_value in final_score.items():
        metrics["{}_{}_{}".format(dataset_name, split_name, metric_name)] = metric_value
    model.train()
    return metrics


step_cnt = 0
last_train_loss = 0.0
latest_metrics = {}
last_eval_step = None

for epoch_idx in range(1, args.epoch + 1):
    model.train()
    train_losses = []
    train_loop = tqdm(train_dataloader)
    for batches in train_loop:
        step_cnt += 1

        for key in batches.keys():
            batches[key] = batches[key].to(device)
        batches["interpolate_pos_encoding"] = True

        out = model(**batches)
        out.loss.backward()
        train_losses.append(out.loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        train_loop.set_description("loss=" + str(out.loss.item()))

        if step_cnt % eval_step == 0:
            dev_metrics = evaluate(val_dataloader, "dev")
            test_metrics = evaluate(test_dataloader, "test")
            latest_metrics = {
                "{}_train_loss".format(dataset_name): sum(train_losses) / len(train_losses),
                "epoch": epoch_idx,
                "step": step_cnt,
                "lr": optimizer.param_groups[0]["lr"],
            }
            latest_metrics.update(dev_metrics)
            latest_metrics.update(test_metrics)

            if save_path is not None:
                model_name = "model_" + str(step_cnt) + ".bin"
                torch.save(model.state_dict(), os.path.join(save_path, model_name))
            last_eval_step = step_cnt

    if train_losses:
        last_train_loss = sum(train_losses) / len(train_losses)
        print("train_loss = {}".format(last_train_loss))

if last_eval_step != step_cnt:
    dev_metrics = evaluate(val_dataloader, "dev")
    test_metrics = evaluate(test_dataloader, "test")
    latest_metrics = {
        "{}_train_loss".format(dataset_name): last_train_loss,
        "epoch": args.epoch,
        "step": step_cnt,
        "lr": optimizer.param_groups[0]["lr"],
    }
    latest_metrics.update(dev_metrics)
    latest_metrics.update(test_metrics)

metrics_path = save_run_metrics(
    args,
    model_name=model_type,
    task_name=dataset_name,
    final_metrics=latest_metrics,
    checkpoint_dir=save_path,
    extra_metadata={
        "script": "test_image_classification.py",
    },
)
print("saved metrics to", metrics_path)
