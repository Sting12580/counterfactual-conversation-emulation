from __future__ import annotations

import argparse

from cce_data.judge_calibration.runner import run_weighted_judge_combination


def _bool_list(value: str) -> list[bool]:
    mapping = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
    result = []
    for item in value.split(","):
        key = item.strip().lower()
        if key:
            if key not in mapping:
                raise argparse.ArgumentTypeError(f"Invalid boolean value {item!r}")
            result.append(mapping[key])
    if not result:
        raise argparse.ArgumentTypeError("At least one boolean value is required")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run weighted judge combination experiments.")
    parser.add_argument("--score-matrix", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--split-protocol",
        choices=[
            "grouped_random",
            "leave_one_dataset_out",
            "leave_one_score_dimension_out",
            "leave_one_answer_source_out",
        ],
        default="grouped_random",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--lambda-grid", default="0,0.0001,0.001,0.01,0.1,1.0")
    parser.add_argument("--fit-intercept-options", type=_bool_list, default=[True, False])
    parser.add_argument("--group-balance", action="store_true")
    parser.add_argument("--group-balance-col")
    parser.add_argument("--clip-predictions", dest="clip_predictions", action="store_true", default=True)
    parser.add_argument("--no-clip-predictions", dest="clip_predictions", action="store_false")
    parser.add_argument("--judge-prefix", default="judge::")
    parser.add_argument("--min-complete-judge-coverage", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_weighted_judge_combination(
        score_matrix=args.score_matrix,
        out_dir=args.out_dir,
        split_protocol=args.split_protocol,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        lambda_grid=args.lambda_grid,
        fit_intercept_options=args.fit_intercept_options,
        group_balance=args.group_balance,
        group_balance_col=args.group_balance_col,
        clip_predictions=args.clip_predictions,
        judge_prefix=args.judge_prefix,
        min_complete_judge_coverage=args.min_complete_judge_coverage,
    )
    print(f"wrote outputs to {summary['out_dir']}")
    print(f"split protocol: {summary['split_protocol']} ({summary['n_splits']} split(s))")


if __name__ == "__main__":
    main()
