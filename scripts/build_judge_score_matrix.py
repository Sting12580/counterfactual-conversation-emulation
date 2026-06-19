from __future__ import annotations

import argparse
from pathlib import Path

from cce_data.judge_calibration.score_matrix import (
    build_score_matrix,
    generate_mock_judge_scores,
    load_calibration_examples,
    load_judge_scores,
    write_judge_scores_long,
    write_missingness_report,
    write_score_matrix_csv,
)


def _csv_list(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a donor judge score matrix.")
    parser.add_argument("--examples", required=True)
    parser.add_argument("--judge-scores", action="append", default=[])
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--missingness-out", required=True)
    parser.add_argument("--required-judges")
    parser.add_argument("--min-judge-coverage", type=float)
    parser.add_argument("--mock-judges")
    parser.add_argument("--mock-seed", type=int, default=0)
    parser.add_argument("--mock-noise-std", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_calibration_examples(args.examples)
    mock_judges = _csv_list(args.mock_judges)
    judge_scores = []
    if args.judge_scores:
        judge_scores.extend(load_judge_scores(args.judge_scores))
    if mock_judges:
        judge_scores = generate_mock_judge_scores(
            examples,
            judge_ids=mock_judges,
            seed=args.mock_seed,
            noise_std=args.mock_noise_std,
        )
        mock_long_path = Path(args.out_csv).with_suffix(".mock_judge_scores_long.jsonl")
        write_judge_scores_long(mock_long_path, judge_scores)
    if not judge_scores:
        raise ValueError("Provide --judge-scores or --mock-judges")

    df, report = build_score_matrix(
        examples,
        judge_scores,
        required_judges=_csv_list(args.required_judges),
        min_judge_coverage=args.min_judge_coverage,
    )
    if mock_judges:
        report["mock_scores"] = True
        report["mock_seed"] = args.mock_seed
        report["mock_noise_std"] = args.mock_noise_std
    write_score_matrix_csv(df, args.out_csv)
    write_missingness_report(report, args.missingness_out)
    print(f"wrote {len(df)} rows to {args.out_csv}")
    print(f"donor judges: {report['donor_judge_ids']}")
    print(f"complete coverage examples: {report['examples_with_complete_coverage']}")


if __name__ == "__main__":
    main()
