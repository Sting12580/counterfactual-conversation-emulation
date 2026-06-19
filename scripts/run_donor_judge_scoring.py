from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

from cce_data.judge_calibration.judge_clients import make_judge_client
from cce_data.judge_calibration.judge_prompting import build_judge_prompt, parse_score_json
from cce_data.judge_calibration.schema import (
    JudgeScoreRecord,
    calibration_example_from_dict,
    read_jsonl,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score calibration examples with donor judges.")
    parser.add_argument("--examples", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--judge-id", required=True)
    parser.add_argument("--rubric-id", default="domain_dimension_v2")
    parser.add_argument("--provider", default="mock")
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--cache-dir", default="data/judge_calibration/judge_cache")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--parse-retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args()


def _score_with_parse_retries(
    client,
    prompt: str,
    parse_retries: int,
    sleep_seconds: float,
) -> tuple[str, float, dict]:
    last_error: Exception | None = None
    raw_output = ""
    for attempt in range(parse_retries + 1):
        raw_output = client.score(prompt)
        try:
            score, payload = parse_score_json(raw_output)
            return raw_output, score, payload
        except ValueError as exc:
            last_error = exc
            if attempt < parse_retries:
                time.sleep(max(1.0, sleep_seconds))
    raise ValueError(f"Could not parse judge output after retries: {last_error}") from last_error


def main() -> None:
    args = parse_args()
    examples = [calibration_example_from_dict(record) for record in read_jsonl(args.examples)]
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.end_index is not None and args.end_index < args.start_index:
        raise ValueError("--end-index must be >= --start-index")
    examples = examples[args.start_index : args.end_index]
    if args.limit is not None:
        examples = examples[: args.limit]

    dry_run = args.dry_run and not args.execute
    if dry_run:
        for example in examples[:3]:
            prompt, prompt_hash = build_judge_prompt(example, rubric_id=args.rubric_id)
            print(f"dry-run prompt_hash={prompt_hash} calib_example_id={example.calib_example_id}")
            print(prompt[:500])
        print("dry-run only; pass --execute to write scores")
        return

    client = make_judge_client(
        args.provider,
        model=args.model,
        seed=args.seed,
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=args.max_retries,
        max_tokens=args.max_tokens,
    )
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    records: list[JudgeScoreRecord] = []
    for index, example in enumerate(examples, start=1):
        prompt, prompt_hash = build_judge_prompt(example, rubric_id=args.rubric_id)
        cache_path = cache_dir / f"{example.calib_example_id}__{args.judge_id}__{prompt_hash}.json"
        if cache_path.exists() and not args.refresh_cache:
            raw_output = cache_path.read_text(encoding="utf-8")
            try:
                score, payload = parse_score_json(raw_output)
            except ValueError:
                raw_output, score, payload = _score_with_parse_retries(
                    client,
                    prompt,
                    parse_retries=args.parse_retries,
                    sleep_seconds=args.sleep_seconds,
                )
                cache_path.write_text(raw_output, encoding="utf-8")
        else:
            raw_output, score, payload = _score_with_parse_retries(
                client,
                prompt,
                parse_retries=args.parse_retries,
                sleep_seconds=args.sleep_seconds,
            )
            cache_path.write_text(raw_output, encoding="utf-8")
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
        records.append(
            JudgeScoreRecord(
                calib_example_id=example.calib_example_id,
                judge_id=args.judge_id,
                score=score,
                score_raw=payload,
                score_scale={"min": 0, "max": 1},
                rubric_id=args.rubric_id,
                prompt_hash=prompt_hash,
                model_version=args.model or args.provider,
                created_at=datetime.now(timezone.utc).isoformat(),
                metadata={"provider": args.provider, "cache_path": str(cache_path)},
            )
        )
        if index % 25 == 0 or index == len(examples):
            print(f"scored {index}/{len(examples)} examples for {args.judge_id}", flush=True)
    write_jsonl(args.out, records)
    print(f"wrote {len(records)} judge score records to {args.out}")


if __name__ == "__main__":
    main()
