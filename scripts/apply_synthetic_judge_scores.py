from __future__ import annotations

import argparse

from cce_data.judge_calibration.apply import (
    apply_predictions_to_paired_files,
    write_answer_level_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply trained synthetic judge weights.")
    parser.add_argument("--canonical-examples", required=True)
    parser.add_argument("--score-matrix", required=True)
    parser.add_argument("--weights-json", required=True)
    parser.add_argument(
        "--out-predictions",
        default="data/judge_calibration/synthetic_judge_predictions.jsonl",
    )
    parser.add_argument("--paired-input", action="append", default=[])
    parser.add_argument("--paired-output-dir")
    parser.add_argument("--overwrite-reward-fields", action="store_true")
    parser.add_argument("--on-missing", choices=["fail", "skip"], default="fail")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = write_answer_level_predictions(
        canonical_examples=args.canonical_examples,
        score_matrix=args.score_matrix,
        weights_json=args.weights_json,
        out_predictions=args.out_predictions,
    )
    print(f"wrote {len(predictions)} answer-level predictions to {args.out_predictions}")
    if args.paired_input:
        if not args.paired_output_dir:
            raise ValueError("--paired-output-dir is required when --paired-input is provided")
        written = apply_predictions_to_paired_files(
            canonical_examples=args.canonical_examples,
            predictions=predictions,
            paired_inputs=args.paired_input,
            paired_output_dir=args.paired_output_dir,
            overwrite_reward_fields=args.overwrite_reward_fields,
            on_missing=args.on_missing,
        )
        for path in written:
            print(f"wrote paired calibrated file: {path}")


if __name__ == "__main__":
    main()
