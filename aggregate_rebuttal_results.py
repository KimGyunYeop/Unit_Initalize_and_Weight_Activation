import argparse
import csv
import json
import math
import os
import statistics

IGNORED_METRIC_NAMES = {"epoch", "step", "steps", "update_steps", "lr"}


def parse_seed_values(raw_seed_args):
    if not raw_seed_args:
        return None

    seed_values = []
    for raw_seed_arg in raw_seed_args:
        for value in raw_seed_arg.split(","):
            stripped_value = value.strip()
            if not stripped_value:
                continue
            seed_values.append(int(stripped_value))
    return sorted(set(seed_values))


def build_parser():
    parser = argparse.ArgumentParser(description="Aggregate rebuttal experiment metrics.")
    parser.add_argument("--input_root", type=str, default="rebuttal_runs")
    parser.add_argument("--output_prefix", type=str, default="aggregated_rebuttal_results")
    parser.add_argument("--baseline_preset", type=str, default="baseline")
    parser.add_argument("--seeds", action="append", default=None)
    return parser


def load_metrics(input_root, selected_seeds=None):
    loaded_runs = []
    for root, _, files in os.walk(input_root):
        if "metrics.json" not in files:
            continue

        metrics_path = os.path.join(root, "metrics.json")
        with open(metrics_path, "r", encoding="utf-8") as metrics_file:
            payload = json.load(metrics_file)

        if selected_seeds is not None and payload.get("seed") not in selected_seeds:
            continue
        loaded_runs.append(payload)
    return loaded_runs


def is_numeric_metric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def should_aggregate_metric(metric_name, metric_value):
    if metric_name in IGNORED_METRIC_NAMES:
        return False
    return is_numeric_metric(metric_value)


def aggregate_runs(runs, baseline_preset):
    grouped_values = {}
    baseline_values = {}

    for run in runs:
        final_metrics = run.get("final_metrics", {})
        group_key = (
            run.get("model"),
            run.get("task"),
            run.get("preset"),
            run.get("result_path"),
            run.get("init_type"),
            run.get("activation_kind"),
            run.get("activation_placement"),
            run.get("add_position"),
            run.get("add_linear_num"),
            json.dumps(run.get("add_linear_layer"), sort_keys=True),
        )
        baseline_key = (run.get("model"), run.get("task"))

        for metric_name, metric_value in final_metrics.items():
            if not should_aggregate_metric(metric_name, metric_value):
                continue
            grouped_values.setdefault((group_key, metric_name), []).append((run.get("seed"), float(metric_value)))
            if run.get("preset") == baseline_preset:
                baseline_values.setdefault((baseline_key, metric_name), []).append(float(metric_value))

    rows = []
    for (group_key, metric_name), seed_and_values in sorted(grouped_values.items()):
        sorted_seed_and_values = sorted(seed_and_values, key=lambda item: item[0])
        seeds = [seed for seed, _ in sorted_seed_and_values]
        values = [value for _, value in sorted_seed_and_values]
        mean_value = statistics.mean(values)
        std_value = statistics.stdev(values) if len(values) > 1 else 0.0

        model_name, task_name, preset, result_path, init_type, activation_kind, activation_placement, add_position, add_linear_num, add_linear_layer = group_key
        baseline_key = ((model_name, task_name), metric_name)
        baseline_mean = None
        baseline_delta = None
        retention_ratio = None
        if baseline_key in baseline_values:
            baseline_mean = statistics.mean(baseline_values[baseline_key])
            baseline_delta = mean_value - baseline_mean
            if baseline_mean != 0:
                retention_ratio = mean_value / baseline_mean

        rows.append(
            {
                "model": model_name,
                "task": task_name,
                "preset": preset,
                "result_path": result_path,
                "init_type": init_type,
                "activation_kind": activation_kind,
                "activation_placement": activation_placement,
                "add_position": add_position,
                "add_linear_num": add_linear_num,
                "add_linear_layer": json.loads(add_linear_layer) if add_linear_layer not in ["null", "\"None\""] else None,
                "metric_name": metric_name,
                "num_seeds": len(values),
                "seeds": seeds,
                "mean": mean_value,
                "std": std_value,
                "baseline_mean": baseline_mean,
                "delta_vs_baseline": baseline_delta,
                "retention_ratio_vs_no_added_layer": retention_ratio,
                "values_by_seed": sorted_seed_and_values,
            }
        )
    return rows


def write_csv(rows, path):
    if not rows:
        fieldnames = ["model", "task", "preset", "metric_name", "mean", "std"]
    else:
        fieldnames = [
            "model",
            "task",
            "preset",
            "result_path",
            "init_type",
            "activation_kind",
            "activation_placement",
            "add_position",
            "add_linear_num",
            "add_linear_layer",
            "metric_name",
            "num_seeds",
            "seeds",
            "mean",
            "std",
            "baseline_mean",
            "delta_vs_baseline",
            "retention_ratio_vs_no_added_layer",
            "values_by_seed",
        ]

    with open(path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["seeds"] = ",".join(str(seed) for seed in row.get("seeds", []))
            csv_row["values_by_seed"] = json.dumps(row.get("values_by_seed", []))
            writer.writerow(csv_row)


def write_json(rows, raw_runs, path):
    payload = {
        "num_runs": len(raw_runs),
        "num_aggregated_rows": len(rows),
        "aggregated_rows": rows,
        "raw_runs": raw_runs,
    }
    with open(path, "w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, sort_keys=True)


def format_value(value):
    if value is None:
        return "-"
    return "{:.6f}".format(value)


def write_markdown(rows, path):
    header = [
        "| model | task | preset | metric | n | mean | std | baseline | delta | retention |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    body = []
    for row in rows:
        body.append(
            "| {model} | {task} | {preset} | {metric_name} | {num_seeds} | {mean} | {std} | {baseline} | {delta} | {retention} |".format(
                model=row["model"],
                task=row["task"],
                preset=row["preset"],
                metric_name=row["metric_name"],
                num_seeds=row["num_seeds"],
                mean=format_value(row["mean"]),
                std=format_value(row["std"]),
                baseline=format_value(row["baseline_mean"]),
                delta=format_value(row["delta_vs_baseline"]),
                retention=format_value(row["retention_ratio_vs_no_added_layer"]),
            )
        )
    with open(path, "w", encoding="utf-8") as markdown_file:
        markdown_file.write("\n".join(header + body) + "\n")


def main():
    parser = build_parser()
    args = parser.parse_args()

    selected_seeds = parse_seed_values(args.seeds)
    runs = load_metrics(args.input_root, selected_seeds=selected_seeds)
    rows = aggregate_runs(runs, baseline_preset=args.baseline_preset)

    csv_path = args.output_prefix + ".csv"
    json_path = args.output_prefix + ".json"
    markdown_path = args.output_prefix + ".md"

    write_csv(rows, csv_path)
    write_json(rows, runs, json_path)
    write_markdown(rows, markdown_path)

    print("loaded runs:", len(runs))
    print("aggregated rows:", len(rows))
    print("csv:", csv_path)
    print("json:", json_path)
    print("markdown:", markdown_path)


if __name__ == "__main__":
    main()
