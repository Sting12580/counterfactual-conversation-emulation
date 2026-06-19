from __future__ import annotations

import argparse

from cce_data.judge_calibration.canonicalize import build_canonical_dataset, write_canonical_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build answer-level judge calibration examples.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--out",
        default="data/judge_calibration/canonical_examples_phase7_v2.jsonl",
    )
    parser.add_argument(
        "--summary-out",
        default="outputs/judge_calibration/data_summary_phase7_v2.json",
    )
    parser.add_argument("--include-ayers", dest="include_ayers", action="store_true", default=True)
    parser.add_argument("--no-include-ayers", dest="include_ayers", action="store_false")
    parser.add_argument(
        "--ayers-dimensions",
        default="composite",
        help="Comma-separated Ayers dimensions to include, e.g. composite or composite,quality,empathy.",
    )
    parser.add_argument(
        "--include-counselbench", dest="include_counselbench", action="store_true", default=True
    )
    parser.add_argument("--no-include-counselbench", dest="include_counselbench", action="store_false")
    parser.add_argument(
        "--include-mediqa-qa-2019",
        dest="include_mediqa_qa_2019",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-include-mediqa-qa-2019",
        dest="include_mediqa_qa_2019",
        action="store_false",
    )
    parser.add_argument(
        "--include-liveqa-medical-2017",
        dest="include_liveqa_medical_2017",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-include-liveqa-medical-2017",
        dest="include_liveqa_medical_2017",
        action="store_false",
    )
    parser.add_argument("--mediqa-qa-2019-path")
    parser.add_argument("--liveqa-medical-2017-path")
    parser.add_argument("--fail-on-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples, summary = build_canonical_dataset(
        repo_root=args.repo_root,
        include_ayers=args.include_ayers,
        ayers_dimensions=args.ayers_dimensions,
        include_counselbench=args.include_counselbench,
        include_mediqa_qa_2019=args.include_mediqa_qa_2019,
        include_liveqa_medical_2017=args.include_liveqa_medical_2017,
        mediqa_qa_2019_path=args.mediqa_qa_2019_path,
        liveqa_medical_2017_path=args.liveqa_medical_2017_path,
        fail_on_missing=args.fail_on_missing,
    )
    write_canonical_outputs(examples, args.out, args.summary_out, summary=summary)
    print(f"total examples: {summary['total_examples']}")
    print(f"by dataset_id: {summary['by_dataset_id']}")
    print(f"by score_dimension: {summary['by_score_dimension']}")
    print(f"by answer_source: {summary['by_answer_source']}")
    print(f"human_score min/max: {summary['human_score_min']} / {summary['human_score_max']}")
    print(f"duplicates removed: {summary.get('duplicates_removed', 0)}")
    print(f"skipped examples: {summary.get('skipped_examples', 0)}")
    print(f"skipped by dataset_id: {summary.get('skipped_by_dataset_id', {})}")
    print(f"skipped reasons by dataset_id: {summary.get('skipped_reasons_by_dataset_id', {})}")
    if summary.get("missing_files"):
        print(f"missing files: {summary['missing_files']}")


if __name__ == "__main__":
    main()
