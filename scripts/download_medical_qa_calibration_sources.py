from __future__ import annotations

import argparse
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

import requests


MEDIQA_BASE_URL = (
    "https://raw.githubusercontent.com/abachaa/MEDIQA2019/master/MEDIQA_Task3_QA"
)
MEDIQA_FILES = {
    "train_liveqa_med": "MEDIQA2019-Task3-QA-TrainingSet1-LiveQAMed.xml",
    "train_alexa": "MEDIQA2019-Task3-QA-TrainingSet2-Alexa.xml",
    "validation": "MEDIQA2019-Task3-QA-ValidationSet.xml",
    "test": "MEDIQA2019-Task3-QA-TestSet-wLabels.xml",
}
MEDIQA_README = "Task3_README.txt"

LIVEQA_BASE_URL = (
    "https://raw.githubusercontent.com/abachaa/LiveQA_MedicalTask_TREC2017/master"
)
LIVEQA_TEST_XML = "TestDataset/TREC-2017-LiveQA-Medical-Test.xml"
LIVEQA_QRELS = "TestDataset/TREC-2017-LiveQA-Medical-qrels-NIST-692.txt"
LIVEQA_README = "Readme.txt"


def _collapse_ws(text: str | None) -> str:
    return " ".join((text or "").split())


def _download(url: str, dest: Path, *, force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _safe_int(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _download_mediqa(raw_dir: Path, *, force: bool) -> dict[str, Path]:
    mediqa_raw_dir = raw_dir / "mediqa_qa_2019"
    paths: dict[str, Path] = {}
    for split, filename in MEDIQA_FILES.items():
        paths[split] = _download(
            f"{MEDIQA_BASE_URL}/{filename}",
            mediqa_raw_dir / filename,
            force=force,
        )
    _download(f"{MEDIQA_BASE_URL}/{MEDIQA_README}", mediqa_raw_dir / MEDIQA_README, force=force)
    return paths


def _convert_mediqa(paths_by_split: dict[str, Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, path in paths_by_split.items():
        root = ET.parse(path).getroot()
        for question in root.findall(".//Question"):
            source_qid = question.get("QID") or ""
            question_text = _collapse_ws(question.findtext("QuestionText"))
            if not question_text:
                continue
            for answer_index, answer in enumerate(question.findall("./AnswerList/Answer")):
                answer_text = _collapse_ws(answer.findtext("AnswerText"))
                if not answer_text:
                    continue
                source_aid = answer.get("AID") or f"{source_qid}_answer_{answer_index}"
                row = {
                    "question_id": f"{split}:{source_qid}",
                    "qid": source_qid,
                    "answer_id": source_aid,
                    "question": question_text,
                    "answer": answer_text,
                    "answer_source": "chiqa",
                    "raw_score": _safe_int(answer.get("ReferenceScore")),
                    "system_rank": _safe_int(answer.get("SystemRank")),
                    "reference_rank": _safe_int(answer.get("ReferenceRank")),
                    "answer_url": _collapse_ws(answer.findtext("AnswerURL")),
                    "category": split,
                    "source_dataset": "MEDIQA-QA 2019",
                    "source_file": path.name,
                }
                rows.append(row)
    return rows


def _download_liveqa(raw_dir: Path, *, force: bool) -> tuple[Path, Path]:
    liveqa_raw_dir = raw_dir / "liveqa_medical_2017"
    test_xml = _download(
        f"{LIVEQA_BASE_URL}/{LIVEQA_TEST_XML}",
        liveqa_raw_dir / Path(LIVEQA_TEST_XML).name,
        force=force,
    )
    qrels = _download(
        f"{LIVEQA_BASE_URL}/{LIVEQA_QRELS}",
        liveqa_raw_dir / Path(LIVEQA_QRELS).name,
        force=force,
    )
    _download(f"{LIVEQA_BASE_URL}/{LIVEQA_README}", liveqa_raw_dir / LIVEQA_README, force=force)
    return test_xml, qrels


def _liveqa_question_map(test_xml: Path) -> dict[str, dict[str, str]]:
    root = ET.parse(test_xml).getroot()
    questions: dict[str, dict[str, str]] = {}
    for question in root.findall(".//NLM-QUESTION"):
        qid = question.get("qid") or ""
        subject = _collapse_ws(question.findtext(".//SUBJECT"))
        message = _collapse_ws(question.findtext(".//MESSAGE"))
        paraphrase = _collapse_ws(question.findtext("NIST-PARAPHRASE"))
        summary = _collapse_ws(question.findtext("NLM-Summary"))
        types = [_collapse_ws("".join(item.itertext())) for item in question.findall(".//TYPE")]
        foci = [_collapse_ws("".join(item.itertext())) for item in question.findall(".//FOCUS")]
        original = ". ".join(part for part in [subject, message] if part)
        questions[qid] = {
            "question": paraphrase or summary or original,
            "topic": subject,
            "description": original if paraphrase and original != paraphrase else "",
            "category": ",".join(item for item in types if item),
            "focus": ",".join(item for item in foci if item),
        }
    return questions


def _convert_liveqa(test_xml: Path, qrels: Path) -> tuple[list[dict[str, Any]], int]:
    questions = _liveqa_question_map(test_xml)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    duplicate_rows = 0

    for line_number, line in enumerate(qrels.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        qid_token, raw_grade, answer_text = line.split(maxsplit=2)
        answer_text = _collapse_ws(answer_text)
        dedupe_key = (qid_token, raw_grade, answer_text)
        if dedupe_key in seen:
            duplicate_rows += 1
            continue
        seen.add(dedupe_key)

        question_id = f"TQ{qid_token}"
        question_info = questions.get(question_id, {})
        row = {
            "question_id": question_id,
            "qid": question_id,
            "question": question_info.get("question") or question_id,
            "answer": answer_text,
            "answer_source": "trec_liveqa_judged_answer",
            "raw_grade": _safe_int(raw_grade),
            "category": question_info.get("category", ""),
            "topic": question_info.get("topic", ""),
            "description": question_info.get("description", ""),
            "focus": question_info.get("focus", ""),
            "source_dataset": "TREC LiveQA Medical 2017",
            "source_file": qrels.name,
            "source_line": line_number,
        }
        rows.append(row)
    return rows, duplicate_rows


def _counts(rows: list[dict[str, Any]], score_field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(score_field)) for row in rows).items()))


def _norm_dedupe_text(text: str | None) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text.casefold()


def _dedupe_answer_rows(
    rows: list[dict[str, Any]], score_field: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: OrderedDict[tuple[str, str], list[dict[str, Any]]] = OrderedDict()
    for row in rows:
        key = (_norm_dedupe_text(row.get("question")), _norm_dedupe_text(row.get("answer")))
        grouped.setdefault(key, []).append(row)

    kept: list[dict[str, Any]] = []
    exact_duplicate_rows_removed = 0
    conflicting_duplicate_groups = 0
    conflicting_duplicate_rows_removed = 0

    for group in grouped.values():
        if len(group) == 1:
            kept.append(group[0])
            continue
        scores = {row.get(score_field) for row in group}
        if len(scores) == 1:
            kept.append(group[0])
            exact_duplicate_rows_removed += len(group) - 1
            continue
        conflicting_duplicate_groups += 1
        conflicting_duplicate_rows_removed += len(group)

    return kept, {
        "answer_pair_duplicate_groups": sum(1 for group in grouped.values() if len(group) > 1),
        "answer_pair_exact_duplicate_rows_removed": exact_duplicate_rows_removed,
        "answer_pair_conflicting_duplicate_groups_removed": conflicting_duplicate_groups,
        "answer_pair_conflicting_duplicate_rows_removed": conflicting_duplicate_rows_removed,
    }


def _filter_mediqa_rows(
    rows: list[dict[str, Any]],
    *,
    drop_liveqa_overlap: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report: dict[str, Any] = {"input_rows": len(rows)}
    if drop_liveqa_overlap:
        before = len(rows)
        rows = [row for row in rows if row.get("category") != "train_liveqa_med"]
        report["train_liveqa_med_overlap_rows_removed"] = before - len(rows)
    else:
        report["train_liveqa_med_overlap_rows_removed"] = 0

    rows, dedupe_report = _dedupe_answer_rows(rows, "raw_score")
    report.update(dedupe_report)
    report["output_rows"] = len(rows)
    return rows, report


def _filter_liveqa_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report: dict[str, Any] = {"input_rows": len(rows)}
    before = len(rows)
    rows = [row for row in rows if row.get("raw_grade") in {1, 2, 3, 4}]
    report["unusable_raw_grade_rows_removed"] = before - len(rows)

    rows, dedupe_report = _dedupe_answer_rows(rows, "raw_grade")
    report.update(dedupe_report)
    report["output_rows"] = len(rows)
    return rows, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and convert MEDIQA-QA 2019 and TREC LiveQA Medical 2017."
    )
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-root", default="data")
    parser.add_argument("--manifest", default="data/medical_qa_calibration_sources_manifest.json")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--sources",
        default="mediqa_qa_2019,liveqa_medical_2017",
        help="Comma-separated subset: mediqa_qa_2019,liveqa_medical_2017.",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help=(
            "Keep all converted rows. By default, the script filters duplicate answer pairs, "
            "conflicting duplicate labels, unusable LiveQA grades, and MEDIQA's LiveQA overlap "
            "when both datasets are selected."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_root = Path(args.output_root)
    selected = {item.strip() for item in args.sources.split(",") if item.strip()}
    unknown = selected - {"mediqa_qa_2019", "liveqa_medical_2017"}
    if unknown:
        raise ValueError(f"Unknown source(s): {sorted(unknown)}")

    manifest: dict[str, Any] = {"filtering_enabled": not args.no_filter}

    if "mediqa_qa_2019" in selected:
        mediqa_paths = _download_mediqa(raw_dir, force=args.force)
        mediqa_rows = _convert_mediqa(mediqa_paths)
        if not args.no_filter:
            mediqa_rows, filter_report = _filter_mediqa_rows(
                mediqa_rows,
                drop_liveqa_overlap="liveqa_medical_2017" in selected,
            )
        else:
            filter_report = {"input_rows": len(mediqa_rows), "output_rows": len(mediqa_rows)}
        mediqa_out = output_root / "mediqa_qa_2019" / "answers.jsonl"
        _write_jsonl(mediqa_out, mediqa_rows)
        manifest["mediqa_qa_2019"] = {
            "output": str(mediqa_out),
            "rows": len(mediqa_rows),
            "raw_score_counts": _counts(mediqa_rows, "raw_score"),
            "raw_files": {split: str(path) for split, path in mediqa_paths.items()},
            "filter_report": filter_report,
        }

    if "liveqa_medical_2017" in selected:
        liveqa_test_xml, liveqa_qrels = _download_liveqa(raw_dir, force=args.force)
        liveqa_rows, duplicate_rows = _convert_liveqa(liveqa_test_xml, liveqa_qrels)
        if not args.no_filter:
            liveqa_rows, filter_report = _filter_liveqa_rows(liveqa_rows)
        else:
            filter_report = {"input_rows": len(liveqa_rows), "output_rows": len(liveqa_rows)}
        liveqa_out = output_root / "liveqa_medical_2017" / "answers.jsonl"
        _write_jsonl(liveqa_out, liveqa_rows)
        manifest["liveqa_medical_2017"] = {
            "output": str(liveqa_out),
            "rows": len(liveqa_rows),
            "raw_grade_counts": _counts(liveqa_rows, "raw_grade"),
            "exact_duplicate_qrels_removed": duplicate_rows,
            "filter_report": filter_report,
            "raw_files": {
                "test_xml": str(liveqa_test_xml),
                "qrels": str(liveqa_qrels),
            },
        }

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
