from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


EXAMPLES = Path("data/judge_calibration/canonical_examples_phase7_v2.jsonl")
PART_DIR = Path("data/judge_calibration/judge_scores/phase7_v2_parts")
FINAL_SCORE_DIR = Path("data/judge_calibration/judge_scores")
LOG_DIR = Path("outputs/judge_calibration/phase7_v2_runner_logs")
MATRIX_PATH = Path("data/judge_calibration/score_matrix_phase7_v2_5judge_gemini25flash_think1024_cohere.csv")
MISSINGNESS_PATH = Path("outputs/judge_calibration/phase7_v2_missingness_report.json")
LAMBDA_GRID = "0,0.0001,0.001,0.01,0.1,1.0"
FIT_INTERCEPT_OPTIONS = "true,false"


@dataclass(frozen=True)
class JudgeConfig:
    judge_id: str
    file_prefix: str
    provider: str
    model: str
    chunk_size: int
    concurrency: int
    max_retries: int
    parse_retries: int
    max_tokens: int = 1024
    sleep_seconds: float = 0.0


JUDGES = [
    JudgeConfig(
        judge_id="openai_gpt55",
        file_prefix="openai_gpt55",
        provider="openai",
        model="gpt-5.5",
        chunk_size=500,
        concurrency=2,
        max_retries=5,
        parse_retries=2,
    ),
    JudgeConfig(
        judge_id="anthropic_sonnet46",
        file_prefix="anthropic_sonnet46",
        provider="anthropic",
        model="claude-sonnet-4-6",
        chunk_size=500,
        concurrency=1,
        max_retries=5,
        parse_retries=2,
    ),
    JudgeConfig(
        judge_id="google_gemini25flash_think1024",
        file_prefix="gemini25flash_think1024",
        provider="gemini",
        model="gemini-2.5-flash",
        chunk_size=500,
        concurrency=1,
        max_retries=8,
        parse_retries=2,
    ),
    JudgeConfig(
        judge_id="cohere_command_a_plus",
        file_prefix="cohere_command_a_plus",
        provider="cohere",
        model="command-a-03-2025",
        chunk_size=100,
        concurrency=1,
        max_retries=5,
        parse_retries=2,
        sleep_seconds=6.0,
    ),
    JudgeConfig(
        judge_id="mistral_large",
        file_prefix="mistral_large",
        provider="mistral",
        model="mistral-large-latest",
        chunk_size=100,
        concurrency=1,
        max_retries=8,
        parse_retries=2,
        sleep_seconds=12.0,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 7 v2 judge calibration pipeline.")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--skip-scoring", action="store_true")
    parser.add_argument(
        "--only-judges",
        help="Comma-separated judge IDs to score. Requires --score-only.",
    )
    parser.add_argument(
        "--exclude-judges",
        help="Comma-separated judge IDs to exclude from scoring. Requires --score-only.",
    )
    return parser.parse_args()


def parse_judge_id_set(raw: str | None) -> set[str]:
    if raw is None:
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


def selected_judges(args: argparse.Namespace) -> list[JudgeConfig]:
    only = parse_judge_id_set(args.only_judges)
    excluded = parse_judge_id_set(args.exclude_judges)
    known = {judge.judge_id for judge in JUDGES}
    unknown = (only | excluded) - known
    if unknown:
        raise ValueError(f"Unknown judge IDs: {', '.join(sorted(unknown))}")
    overlap = only & excluded
    if overlap:
        raise ValueError(f"Judge IDs cannot be both included and excluded: {', '.join(sorted(overlap))}")
    if (only or excluded) and not args.score_only:
        raise ValueError("--only-judges and --exclude-judges require --score-only")
    if only:
        return [judge for judge in JUDGES if judge.judge_id in only]
    if excluded:
        return [judge for judge in JUDGES if judge.judge_id not in excluded]
    return JUDGES


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def total_examples() -> int:
    return count_lines(EXAMPLES)


def chunk_ranges(total: int, chunk_size: int) -> list[tuple[int, int]]:
    return [(start, min(start + chunk_size, total)) for start in range(0, total, chunk_size)]


def part_path(judge: JudgeConfig, start: int, end: int) -> Path:
    return PART_DIR / f"{judge.file_prefix}_part_{start:04d}_{end:04d}.jsonl"


def expected_count(start: int, end: int) -> int:
    return end - start


def is_complete(path: Path, start: int, end: int) -> bool:
    return count_lines(path) == expected_count(start, end)


def command_contains_active_part(command: str, path: Path) -> bool:
    return "run_donor_judge_scoring.py" in command and str(path) in command


def active_part_paths() -> set[Path]:
    result = subprocess.run(
        ["ps", "-ww", "-axo", "pid=,command="],
        check=True,
        capture_output=True,
        text=True,
    )
    paths: set[Path] = set()
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2:
            continue
        command = fields[1]
        executable = Path(command.split()[0]).name
        if "python" not in executable:
            continue
        if "scripts/run_donor_judge_scoring.py" not in command:
            continue
        if "phase7_v2_parts" not in command:
            continue
        for token in command.split():
            if "data/judge_calibration/judge_scores/phase7_v2_parts/" in token:
                paths.add(Path(token))
    return paths


def score_command(judge: JudgeConfig, start: int, end: int, out_path: Path) -> list[str]:
    command = [
        sys.executable,
        "scripts/run_donor_judge_scoring.py",
        "--examples",
        str(EXAMPLES),
        "--out",
        str(out_path),
        "--judge-id",
        judge.judge_id,
        "--provider",
        judge.provider,
        "--model",
        judge.model,
        "--rubric-id",
        "domain_dimension_v2",
        "--start-index",
        str(start),
        "--end-index",
        str(end),
        "--execute",
        "--timeout",
        "180",
        "--max-retries",
        str(judge.max_retries),
        "--parse-retries",
        str(judge.parse_retries),
        "--max-tokens",
        str(judge.max_tokens),
    ]
    if judge.sleep_seconds:
        command.extend(["--sleep-seconds", str(judge.sleep_seconds)])
    return command


def env_with_pythonpath() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = "src" if not existing else f"src{os.pathsep}{existing}"
    return env


def start_score_process(judge: JudgeConfig, start: int, end: int) -> subprocess.Popen[bytes]:
    PART_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = part_path(judge, start, end)
    log_path = LOG_DIR / f"{judge.file_prefix}_part_{start:04d}_{end:04d}.log"
    log_handle = log_path.open("ab")
    command = score_command(judge, start, end, out_path)
    print(f"starting {judge.judge_id} {start}-{end}; log={log_path}", flush=True)
    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env_with_pythonpath(),
    )
    process._phase7_log_handle = log_handle  # type: ignore[attr-defined]
    return process


def close_process_log(process: subprocess.Popen[bytes]) -> None:
    handle = getattr(process, "_phase7_log_handle", None)
    if handle is not None:
        handle.close()


def completed_count(judge: JudgeConfig, total: int) -> int:
    count = 0
    for start, end in chunk_ranges(total, judge.chunk_size):
        path = part_path(judge, start, end)
        if is_complete(path, start, end):
            count += expected_count(start, end)
    return count


def print_status(total: int, active: set[Path], judges: list[JudgeConfig]) -> None:
    chunks = []
    for judge in judges:
        done = completed_count(judge, total)
        running = sum(
            1
            for start, end in chunk_ranges(total, judge.chunk_size)
            if part_path(judge, start, end) in active
        )
        chunks.append(f"{judge.judge_id}: {done}/{total}, active={running}")
    print("status | " + " | ".join(chunks), flush=True)


def run_scoring(args: argparse.Namespace, judges: list[JudgeConfig]) -> None:
    total = total_examples()
    if total <= 0:
        raise RuntimeError(f"No examples found in {EXAMPLES}")
    attempts: dict[Path, int] = {}
    managed: dict[Path, subprocess.Popen[bytes]] = {}
    last_status = 0.0

    while True:
        for path, process in list(managed.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            close_process_log(process)
            del managed[path]
            if return_code != 0:
                print(f"process failed for {path} with exit code {return_code}", flush=True)

        active = active_part_paths() | {path for path, process in managed.items() if process.poll() is None}
        if time.time() - last_status >= 60:
            print_status(total, active, judges)
            last_status = time.time()

        all_complete = True
        for judge in judges:
            ranges = chunk_ranges(total, judge.chunk_size)
            judge_active = {
                part_path(judge, start, end)
                for start, end in ranges
                if part_path(judge, start, end) in active
            }
            for start, end in ranges:
                path = part_path(judge, start, end)
                if is_complete(path, start, end):
                    continue
                all_complete = False
                if path in active:
                    continue
                if len(judge_active) >= judge.concurrency:
                    break
                attempt_count = attempts.get(path, 0)
                if attempt_count >= args.max_attempts:
                    raise RuntimeError(f"Exceeded max attempts for {path}")
                attempts[path] = attempt_count + 1
                process = start_score_process(judge, start, end)
                managed[path] = process
                judge_active.add(path)
                active.add(path)

        if all_complete:
            for process in managed.values():
                close_process_log(process)
            print("all judge scoring chunks are complete", flush=True)
            return

        time.sleep(args.poll_seconds)


def concatenate_judge_scores() -> list[Path]:
    total = total_examples()
    FINAL_SCORE_DIR.mkdir(parents=True, exist_ok=True)
    final_paths: list[Path] = []
    for judge in JUDGES:
        final_path = FINAL_SCORE_DIR / f"{judge.file_prefix}_phase7_v2.jsonl"
        with final_path.open("wb") as out_handle:
            for start, end in chunk_ranges(total, judge.chunk_size):
                path = part_path(judge, start, end)
                if not is_complete(path, start, end):
                    raise RuntimeError(f"Incomplete part before concatenation: {path}")
                with path.open("rb") as in_handle:
                    out_handle.write(in_handle.read())
        final_paths.append(final_path)
        print(f"wrote {count_lines(final_path)} records to {final_path}", flush=True)
    return final_paths


def run_checked(command: list[str]) -> None:
    print("running: " + " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env_with_pythonpath())


def build_matrix(score_paths: list[Path]) -> None:
    command = [
        sys.executable,
        "scripts/build_judge_score_matrix.py",
        "--examples",
        str(EXAMPLES),
    ]
    for path in score_paths:
        command.extend(["--judge-scores", str(path)])
    command.extend(
        [
            "--out-csv",
            str(MATRIX_PATH),
            "--missingness-out",
            str(MISSINGNESS_PATH),
            "--required-judges",
            ",".join(judge.judge_id for judge in JUDGES),
        ]
    )
    run_checked(command)


def run_weight_experiments() -> None:
    specs = [
        (
            "grouped_random",
            "outputs/judge_calibration/phase7_v2_5judge_gemini25flash_think1024_cohere_grouped_random_balanced_dataset_dimension",
        ),
        (
            "leave_one_dataset_out",
            "outputs/judge_calibration/phase7_v2_5judge_gemini25flash_think1024_cohere_lodo_balanced_dataset_dimension",
        ),
        (
            "leave_one_answer_source_out",
            "outputs/judge_calibration/phase7_v2_5judge_gemini25flash_think1024_cohere_loaso_balanced_dataset_dimension",
        ),
    ]
    for protocol, out_dir in specs:
        run_checked(
            [
                sys.executable,
                "scripts/run_weighted_judge_combination.py",
                "--score-matrix",
                str(MATRIX_PATH),
                "--out-dir",
                out_dir,
                "--split-protocol",
                protocol,
                "--seed",
                "0",
                "--group-balance",
                "--lambda-grid",
                LAMBDA_GRID,
                "--fit-intercept-options",
                FIT_INTERCEPT_OPTIONS,
            ]
        )
    run_checked(
        [
            sys.executable,
            "scripts/run_judge_weight_bootstrap.py",
            "--score-matrix",
            str(MATRIX_PATH),
            "--out-dir",
            "outputs/judge_calibration/phase7_v2_5judge_gemini25flash_think1024_cohere_bootstrap_weights",
            "--n-bootstrap",
            "100",
            "--seed",
            "0",
            "--group-balance",
            "--lambda-grid",
            LAMBDA_GRID,
            "--fit-intercept-options",
            FIT_INTERCEPT_OPTIONS,
        ]
    )


def main() -> None:
    args = parse_args()
    judges = selected_judges(args)
    if not args.skip_scoring:
        run_scoring(args, judges)
    if args.score_only and judges != JUDGES:
        return
    score_paths = concatenate_judge_scores()
    if args.score_only:
        return
    build_matrix(score_paths)
    run_weight_experiments()
    print("phase7 v2 pipeline complete", flush=True)


if __name__ == "__main__":
    main()
