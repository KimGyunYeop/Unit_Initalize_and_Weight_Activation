import argparse
import json
import os
import random
import subprocess
import sys

import numpy as np
import torch

AVAIL_BATCH = {
    "rte": 8
}

REBUTTAL_PRESETS = {
    "baseline": {
        "no_add_linear": True,
        "init_type": "unit",
        "activation_kind": "none",
        "activation_placement": "none",
    },
    "naive_xavier_output_gelu": {
        "no_add_linear": False,
        "init_type": "xavier",
        "activation_kind": "gelu",
        "activation_placement": "output",
    },
    "naive_he_output_relu": {
        "no_add_linear": False,
        "init_type": "he",
        "activation_kind": "relu",
        "activation_placement": "output",
    },
    "xavier_no_activation": {
        "no_add_linear": False,
        "init_type": "xavier",
        "activation_kind": "none",
        "activation_placement": "none",
    },
    "zero_no_activation": {
        "no_add_linear": False,
        "init_type": "zero",
        "activation_kind": "none",
        "activation_placement": "none",
    },
    "unit_no_activation": {
        "no_add_linear": False,
        "init_type": "unit",
        "activation_kind": "none",
        "activation_placement": "none",
    },
    "xavier_weight_gelu": {
        "no_add_linear": False,
        "init_type": "xavier",
        "activation_kind": "gelu",
        "activation_placement": "weight",
    },
    "unit_weight_gelu": {
        "no_add_linear": False,
        "init_type": "unit",
        "activation_kind": "gelu",
        "activation_placement": "weight",
    },
    "unit_weight_relu": {
        "no_add_linear": False,
        "init_type": "unit",
        "activation_kind": "relu",
        "activation_placement": "weight",
    },
}

LEGACY_ACT_TYPE_TO_CONFIG = {
    "midgelu": ("gelu", "weight"),
    "midrelu": ("relu", "weight"),
    "gelu": ("gelu", "output"),
    "relu": ("relu", "output"),
}

CONFIG_TO_LEGACY_ACT_TYPE = {
    ("gelu", "weight"): "midgelu",
    ("relu", "weight"): "midrelu",
    ("gelu", "output"): "gelu",
    ("relu", "output"): "relu",
}


def seed_fix(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _flag_present(argv, flag_name):
    return any(arg == flag_name or arg.startswith(flag_name + "=") for arg in argv)


def _parse_seed_values(raw_seed_args, fallback_seed):
    if not raw_seed_args:
        return [fallback_seed]

    seed_values = []
    for raw_seed_arg in raw_seed_args:
        for value in raw_seed_arg.split(","):
            stripped_value = value.strip()
            if not stripped_value:
                continue
            seed_values.append(int(stripped_value))

    if not seed_values:
        return [fallback_seed]

    ordered_unique_seeds = []
    seen = set()
    for seed in seed_values:
        if seed in seen:
            continue
        seen.add(seed)
        ordered_unique_seeds.append(seed)
    return ordered_unique_seeds


def _apply_preset_defaults(args, explicit_flags):
    if args.preset is None:
        return

    preset_config = REBUTTAL_PRESETS[args.preset]

    if not explicit_flags["no_add_linear"]:
        args.no_add_linear = preset_config["no_add_linear"]
    if not explicit_flags["init_type"]:
        args.init_type = preset_config["init_type"]
    if not explicit_flags["activation_kind"]:
        args.activation_kind = preset_config["activation_kind"]
    if not explicit_flags["activation_placement"]:
        args.activation_placement = preset_config["activation_placement"]
    if not explicit_flags["act_type"]:
        default_act_type = CONFIG_TO_LEGACY_ACT_TYPE.get(
            (preset_config["activation_kind"], preset_config["activation_placement"])
        )
        args.act_type = default_act_type


def _normalize_activation_args(args, explicit_flags):
    has_new_style_activation = (
        explicit_flags["activation_kind"]
        or explicit_flags["activation_placement"]
        or args.activation_kind != "none"
        or args.activation_placement != "none"
    )

    if has_new_style_activation:
        activation_kind = args.activation_kind
        activation_placement = args.activation_placement
    elif args.act_type is not None:
        activation_kind, activation_placement = LEGACY_ACT_TYPE_TO_CONFIG[args.act_type]
    else:
        activation_kind, activation_placement = "none", "none"

    if (activation_kind == "none") != (activation_placement == "none"):
        raise ValueError(
            "activation_kind and activation_placement must both be 'none' or both be non-'none'."
        )

    if activation_kind == "none":
        args.activation_kind = "none"
        args.activation_placement = "none"
        args.act_type = None
        return

    if activation_kind not in ["gelu", "relu"]:
        raise ValueError("activation_kind must be one of ['none', 'gelu', 'relu'].")
    if activation_placement not in ["weight", "output"]:
        raise ValueError("activation_placement must be one of ['none', 'weight', 'output'].")

    args.activation_kind = activation_kind
    args.activation_placement = activation_placement
    args.act_type = CONFIG_TO_LEGACY_ACT_TYPE[(activation_kind, activation_placement)]


def _strip_seed_args(argv):
    stripped_argv = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue

        if arg in ["--seed", "--seeds"]:
            skip_next = True
            continue
        if arg.startswith("--seed=") or arg.startswith("--seeds="):
            continue
        if arg == "--_multi_seed_child":
            continue
        stripped_argv.append(arg)
    return stripped_argv


def normalize_args(args, argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    explicit_flags = {
        "no_add_linear": _flag_present(argv, "--no_add_linear"),
        "init_type": _flag_present(argv, "--init_type"),
        "act_type": _flag_present(argv, "--act_type"),
        "activation_kind": _flag_present(argv, "--activation_kind"),
        "activation_placement": _flag_present(argv, "--activation_placement"),
    }

    args.seed_values = _parse_seed_values(args.seeds, args.seed)
    _apply_preset_defaults(args, explicit_flags)
    _normalize_activation_args(args, explicit_flags)

    if args.no_add_linear and args.preset is None:
        args.preset = "baseline"

    return args


def maybe_dispatch_multi_seed_runs(args):
    if args._multi_seed_child or len(args.seed_values) <= 1:
        args.seed = args.seed_values[0]
        return False

    child_argv = _strip_seed_args(sys.argv[1:])
    script_path = os.path.abspath(sys.argv[0])
    for seed in args.seed_values:
        child_cmd = [
            sys.executable,
            script_path,
            *child_argv,
            "--seed",
            str(seed),
            "--_multi_seed_child",
        ]
        print("[multi-seed] launching seed {}".format(seed))
        subprocess.run(child_cmd, check=True)

    return True


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="various_attention")

    parser.add_argument(
        "--for_cv", default=False, action="store_true"
    )
    parser.add_argument(
        "--result_path", type=str, required=True
    )
    parser.add_argument(
        "--model_load_path", type=str, default=None, required=False
    )
    parser.add_argument(
        "--gpu", type=int, default=0, required=False
    )
    parser.add_argument(
        "--epoch", type=int, default=15, required=False
    )
    parser.add_argument(
        "--gen_train_step", type=int, default=15000, required=False
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, required=False
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-5, required=False
    )
    parser.add_argument(
        "--seed", type=int, default=1234, required=False
    )
    parser.add_argument(
        "--seeds", action="append", default=None,
        help="Comma-separated seed list or repeat the flag multiple times."
    )
    parser.add_argument(
        "--accumulate_step", type=int, default=16, required=False
    )
    parser.add_argument(
        "--data_path", type=str, default="datasets", required=False
    )
    parser.add_argument(
        "--run_output_root", type=str, default="rebuttal_runs", required=False
    )
    parser.add_argument(
        "--skip_checkpoint_save", default=True, action="store_true"
    )

    # optimizer & scheduler detail
    parser.add_argument(
        "--beta1", type=float, default=0.9, required=False
    )
    parser.add_argument(
        "--beta2", type=float, default=0.999, required=False
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, required=False
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, required=False
    )
    parser.add_argument(
        "--eps", type=float, default=1e-6, required=False
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=50, required=False
    )

    # transformer unit init
    parser.add_argument(
        "--no_add_linear", default=False, action="store_true"
    )
    parser.add_argument(
        "--add_linear_num", type=int, default=None
    )
    parser.add_argument(
        "--add_linear_layer", type=str, default=None
    )
    parser.add_argument(
        "--preset", type=str, default=None, choices=list(REBUTTAL_PRESETS.keys())
    )
    parser.add_argument(
        "--init_type", type=str, default="unit", choices=["unit", "xavier", "he", "zero"]
    )
    parser.add_argument(
        "--activation_kind", type=str, default="none", choices=["none", "gelu", "relu"]
    )
    parser.add_argument(
        "--activation_placement", type=str, default="none", choices=["none", "weight", "output"]
    )
    parser.add_argument(
        "--head_indi", default=True, action="store_true"
    )
    parser.add_argument(
        "--add_position", default="befdot", choices=["befdot", "aftffnn1", "aftffnn2", "afterffnn", "both"]
    )
    parser.add_argument(
        "--act_type", type=str, default=None, choices=["midgelu", "gelu", "midrelu", "relu"]
    )
    parser.add_argument(
        "--adapter", default=False, action="store_true"
    )
    parser.add_argument(
        "--diagnostic_mode", type=str, default="none", choices=["none", "relative_perturbation"]
    )

    # glue
    glue_tasks = ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "mnli_matched", "mnli_mismatched", "qnli", "rte", "wnli"]
    parser.add_argument(
        "--glue_task", type=str, default="mrpc", choices=glue_tasks
    )

    # cv-image classification
    image_classification_datasets = ["cifar10", "cifar100", "imagenet-1k"]
    parser.add_argument(
        "--image_classification_dataset", type=str, default="cifar10", choices=image_classification_datasets
    )
    parser.add_argument(
        "--image_size", type=int, default=384
    )
    parser.add_argument(
        "--cifar_eval_step", type=int, default=500
    )
    parser.add_argument(
        "--imagenet_eval_step", type=int, default=50000
    )

    # text_generation
    parser.add_argument(
        "--generation_task", type=str, default="mrpc", choices=["cnndm", "wmt_en_ro"]
    )
    parser.add_argument(
        "--generate_full_batch", type=int, default=128
    )

    parser.add_argument(
        "--logging_step", type=int, default=10000,
    )
    parser.add_argument(
        "--dev", default=False, action="store_true"
    )
    parser.add_argument(
        "--_multi_seed_child", default=False, action="store_true", help=argparse.SUPPRESS
    )

    args = parser.parse_args(argv)
    return normalize_args(args, argv)


def tf_make_result_path(args):
    result_path = [args.result_path]

    if args.for_cv is False:
        try:
            result_path.append(args.task)
        except AttributeError:
            result_path.append(args.glue_task)

    if args.for_cv:
        result_path.append(str(args.learning_rate))

    if args.no_add_linear:
        result_path.append("baseline")
        return "_".join(result_path)

    result_path.append(args.add_position)
    result_path.append(args.init_type)

    if args.add_linear_num is not None:
        if args.add_linear_num > 0:
            result_path.append("bottom" + str(args.add_linear_num))
        else:
            result_path.append("top" + str(args.add_linear_num))

    if args.add_linear_layer is not None:
        result_path.append("layer" + str(args.add_linear_layer))

    if args.head_indi:
        result_path.append("indi")

    if args.act_type is None:
        result_path.append("no_act")
    else:
        result_path.append(args.act_type)

    if args.adapter:
        result_path.append("adapter")

    if args.dev:
        result_path.append("dev")

    return "_".join(result_path)


def gen_make_result_path(args):
    result_path = [args.result_path]

    if args.for_cv is False:
        try:
            result_path.append(args.task)
        except AttributeError:
            result_path.append(args.generation_task)

    if args.for_cv:
        result_path.append(str(args.learning_rate))

    if args.no_add_linear:
        result_path.append("baseline")
        return "_".join(result_path)

    result_path.append(args.add_position)
    result_path.append(args.init_type)

    if args.add_linear_num is not None:
        if args.add_linear_num > 0:
            result_path.append("bottom" + str(args.add_linear_num))
        else:
            result_path.append("top" + str(args.add_linear_num))

    if args.add_linear_layer is not None:
        result_path.append("layer" + str(args.add_linear_layer))

    if args.head_indi:
        result_path.append("indi")

    if args.act_type is None:
        result_path.append("no_act")
    else:
        result_path.append(args.act_type)

    if args.adapter:
        result_path.append("adapter")

    if args.dev:
        result_path.append("dev")

    return "_".join(result_path)


def _sanitize_path_component(value):
    sanitized = str(value).strip().replace(os.sep, "-")
    sanitized = sanitized.replace(" ", "_")
    return sanitized if sanitized else "unknown"


def get_run_output_dir(args, model_name, task_name):
    experiment_name = _sanitize_path_component(args.result_path)
    preset_name = _sanitize_path_component(args.preset or "custom")
    seed_name = "seed_{}".format(args.seed)
    return os.path.join(
        args.run_output_root,
        _sanitize_path_component(model_name),
        _sanitize_path_component(task_name),
        preset_name,
        experiment_name,
        seed_name,
    )


def get_metrics_output_path(args, model_name, task_name):
    return os.path.join(
        get_run_output_dir(args, model_name=model_name, task_name=task_name),
        "metrics.json",
    )


def maybe_skip_completed_run(args, model_name, task_name):
    metrics_path = get_metrics_output_path(args, model_name=model_name, task_name=task_name)
    if os.path.exists(metrics_path):
        print("[skip] existing metrics found at {}".format(metrics_path))
        return True
    return False


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, range):
        return list(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_run_metrics(args, model_name, task_name, final_metrics, checkpoint_dir=None, extra_metadata=None):
    run_output_dir = get_run_output_dir(args, model_name=model_name, task_name=task_name)
    os.makedirs(run_output_dir, exist_ok=True)

    payload = {
        "model": model_name,
        "task": task_name,
        "seed": args.seed,
        "seed_values": args.seed_values,
        "preset": args.preset or ("baseline" if args.no_add_linear else "custom"),
        "result_path": args.result_path,
        "run_output_dir": run_output_dir,
        "checkpoint_dir": checkpoint_dir,
        "no_add_linear": args.no_add_linear,
        "init_type": args.init_type,
        "act_type": args.act_type,
        "activation_kind": args.activation_kind,
        "activation_placement": args.activation_placement,
        "add_position": None if args.no_add_linear else args.add_position,
        "add_linear_num": args.add_linear_num,
        "add_linear_layer": args.add_linear_layer,
        "head_indi": args.head_indi,
        "adapter": args.adapter,
        "skip_checkpoint_save": args.skip_checkpoint_save,
        "final_metrics": final_metrics,
    }
    if extra_metadata:
        payload.update(extra_metadata)

    metrics_path = os.path.join(run_output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(_json_ready(payload), metrics_file, indent=2, sort_keys=True)

    return metrics_path
