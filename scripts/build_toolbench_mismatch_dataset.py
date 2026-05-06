from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "datasets"
DEFAULT_SOURCE_DIR = ROOT / "data" / "raw" / "toolbench" / "G1_answer"
DEFAULT_OUTPUT = DATASET_DIR / "toolbench_mismatch_dataset.csv"

FIELDNAMES = [
    "user_prompt",
    "tool_call_text",
    "label",
    "prompt_source_file",
    "prompt_source_toolkit",
    "tool_source_file",
    "tool_source_toolkit",
    "source_prompt_hash",
    "normal_tool_call_text",
    "mismatch_sampling_type",
]

FINISH_FUNCTION_NAME = "Finish"
EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
URL_RE = re.compile(r"https?://[^\s'\"<>]+|www\.[^\s'\"<>]+", re.IGNORECASE)
UUID_LIKE_TOOL_RE = re.compile(
    r"[0-9a-f]{8}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{12}",
    re.IGNORECASE,
)

EXCLUDED_TOOL_PATTERNS = {
    "newapi": re.compile(r"(^|_)newapi($|_)|_for_newapi$", re.IGNORECASE),
    "demo_project": re.compile(r"demo_project", re.IGNORECASE),
    "onboarding_project": re.compile(r"onboarding_project", re.IGNORECASE),
    "uuid_like_tool": UUID_LIKE_TOOL_RE,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a ToolBench experiment CSV with observed normal pairs and random "
            "mismatch pairs sampled from the same observed ToolBench tool vocabulary."
        )
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Defaults to output path with .summary.json suffix.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timestamp", action="store_true")
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def clean_user_prompt(value: Any) -> str:
    prompt = normalize_text(value)
    if prompt.endswith(" Begin!"):
        prompt = prompt[: -len(" Begin!")].strip()
    return prompt


def mask_public_prompt(prompt: str) -> tuple[str, Counter[str]]:
    counts: Counter[str] = Counter()

    def replace_email(match: re.Match[str]) -> str:
        counts["email_masked"] += 1
        return "[EMAIL]"

    def replace_url(match: re.Match[str]) -> str:
        counts["url_masked"] += 1
        return "[URL]"

    masked = EMAIL_RE.sub(replace_email, prompt)
    masked = URL_RE.sub(replace_url, masked)
    return normalize_text(masked), counts


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def is_successful_payload(payload: dict[str, Any]) -> bool:
    answer_generation = payload.get("answer_generation")
    if not isinstance(answer_generation, dict):
        return False
    return (
        payload.get("win") is True
        and answer_generation.get("valid_data") is True
        and answer_generation.get("finish_type") == "give_answer"
    )


def get_answer_generation(payload: dict[str, Any]) -> dict[str, Any]:
    answer_generation = payload.get("answer_generation")
    if isinstance(answer_generation, dict):
        return answer_generation
    return {}


def get_longest_train_messages(answer_generation: dict[str, Any]) -> list[dict[str, Any]]:
    train_messages = answer_generation.get("train_messages")
    if not isinstance(train_messages, list) or not train_messages:
        return []

    trials = [trial for trial in train_messages if isinstance(trial, list)]
    if not trials:
        return []

    longest = max(trials, key=len)
    return [message for message in longest if isinstance(message, dict)]


def extract_toolkit_from_function(function: dict[str, Any], function_name: str) -> str:
    description = normalize_text(function.get("description"))
    match = re.search(r'tool\s+"([^"]+)"', description)
    if match:
        return normalize_text(match.group(1))

    if "_for_" in function_name:
        return function_name.rsplit("_for_", 1)[1]

    return ""


def get_available_tool_metadata(payload: dict[str, Any]) -> dict[str, dict[str, str]]:
    answer_generation = get_answer_generation(payload)
    functions = answer_generation.get("function")
    if not isinstance(functions, list):
        return {}

    metadata: dict[str, dict[str, str]] = {}
    for function in functions:
        if not isinstance(function, dict):
            continue

        name = normalize_text(function.get("name"))
        if not name or name == FINISH_FUNCTION_NAME:
            continue

        metadata[name] = {
            "toolkit": extract_toolkit_from_function(function, name),
        }
    return metadata


def get_prompt_from_payload(payload: dict[str, Any]) -> str:
    answer_generation = get_answer_generation(payload)
    prompt = clean_user_prompt(answer_generation.get("query"))
    if prompt:
        return prompt

    for message in get_longest_train_messages(answer_generation):
        if message.get("role") == "user":
            return clean_user_prompt(message.get("content"))
    return ""


def extract_observed_tool_calls(payload: dict[str, Any]) -> list[str]:
    answer_generation = get_answer_generation(payload)
    calls: list[str] = []
    seen: set[str] = set()

    for message in get_longest_train_messages(answer_generation):
        if message.get("role") != "assistant":
            continue
        function_call = message.get("function_call")
        if not isinstance(function_call, dict):
            continue

        name = normalize_text(function_call.get("name"))
        if not name or name == FINISH_FUNCTION_NAME or name in seen:
            continue

        seen.add(name)
        calls.append(name)
    return calls


def get_exclusion_reasons(tool_call_text: str) -> list[str]:
    return [
        reason
        for reason, pattern in EXCLUDED_TOOL_PATTERNS.items()
        if pattern.search(tool_call_text)
    ]


def timestamped_path(path: Path, *, timestamp: bool) -> Path:
    resolved = path.resolve()
    if not timestamp:
        return resolved
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = resolved.suffix or ".csv"
    return resolved.with_name(f"{resolved.stem}_{stamp}{suffix}")


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def iter_json_files(source_dir: Path) -> list[Path]:
    return sorted(path for path in source_dir.glob("*.json") if path.is_file())


def build_positive_rows(source_dir: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]], dict[str, Any]]:
    raw_positive_rows: list[dict[str, str]] = []
    tool_catalog: dict[str, dict[str, str]] = {}
    stats: Counter[str] = Counter()
    mask_counts: Counter[str] = Counter()
    tool_quality_removals: Counter[str] = Counter()

    for path in iter_json_files(source_dir):
        stats["json_files_seen"] += 1
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            stats["json_read_errors"] += 1
            continue

        if not isinstance(payload, dict):
            stats["non_object_json"] += 1
            continue

        if not is_successful_payload(payload):
            stats["skipped_not_successful"] += 1
            continue

        prompt = get_prompt_from_payload(payload)
        if not prompt:
            stats["missing_prompt"] += 1
            continue

        public_prompt, row_mask_counts = mask_public_prompt(prompt)
        mask_counts.update(row_mask_counts)
        if not public_prompt:
            stats["empty_prompt_after_masking"] += 1
            continue

        available_tools = get_available_tool_metadata(payload)
        if not available_tools:
            stats["missing_available_tool_list"] += 1
            continue

        observed_calls = extract_observed_tool_calls(payload)
        if not observed_calls:
            stats["missing_observed_tool_calls"] += 1
            continue

        source_file = path.name
        source_prompt_hash = stable_hash(public_prompt)
        kept_for_file = 0

        for tool_call_text in observed_calls:
            if tool_call_text not in available_tools:
                stats["unlisted_observed_tool_calls_removed"] += 1
                continue

            reasons = get_exclusion_reasons(tool_call_text)
            if reasons:
                stats["tool_quality_rows_removed"] += 1
                for reason in reasons:
                    tool_quality_removals[reason] += 1
                continue

            toolkit = available_tools[tool_call_text]["toolkit"]
            row = {
                "user_prompt": public_prompt,
                "tool_call_text": tool_call_text,
                "label": "0",
                "prompt_source_file": source_file,
                "prompt_source_toolkit": toolkit,
                "tool_source_file": source_file,
                "tool_source_toolkit": toolkit,
                "source_prompt_hash": source_prompt_hash,
                "normal_tool_call_text": tool_call_text,
                "mismatch_sampling_type": "observed_normal",
            }
            raw_positive_rows.append(row)
            kept_for_file += 1

            if tool_call_text not in tool_catalog:
                tool_catalog[tool_call_text] = {
                    "tool_source_file": source_file,
                    "tool_source_toolkit": toolkit,
                }

        if kept_for_file:
            stats["source_files_used"] += 1

    positive_rows: list[dict[str, str]] = []
    seen_positive_pairs: set[tuple[str, str]] = set()
    duplicate_positive_rows_removed = 0

    for row in raw_positive_rows:
        key = (row["user_prompt"], row["tool_call_text"])
        if key in seen_positive_pairs:
            duplicate_positive_rows_removed += 1
            continue
        seen_positive_pairs.add(key)
        positive_rows.append(row)

    summary = {
        "json_files_seen": stats["json_files_seen"],
        "json_read_errors": stats["json_read_errors"],
        "non_object_json": stats["non_object_json"],
        "skipped_not_successful": stats["skipped_not_successful"],
        "missing_prompt": stats["missing_prompt"],
        "empty_prompt_after_masking": stats["empty_prompt_after_masking"],
        "missing_available_tool_list": stats["missing_available_tool_list"],
        "missing_observed_tool_calls": stats["missing_observed_tool_calls"],
        "unlisted_observed_tool_calls_removed": stats["unlisted_observed_tool_calls_removed"],
        "tool_quality_rows_removed": stats["tool_quality_rows_removed"],
        "tool_quality_removal_reasons": dict(tool_quality_removals),
        "prompt_mask_counts": dict(mask_counts),
        "source_files_used": stats["source_files_used"],
        "raw_normal_rows": len(raw_positive_rows),
        "duplicate_normal_rows_removed": duplicate_positive_rows_removed,
        "normal_rows": len(positive_rows),
        "tool_catalog_size": len(tool_catalog),
    }
    return positive_rows, tool_catalog, summary


def build_mismatch_rows(
    positive_rows: list[dict[str, str]],
    tool_catalog: dict[str, dict[str, str]],
    *,
    seed: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rng = random.Random(seed)
    all_tools = sorted(tool_catalog)
    positive_tools_by_prompt: dict[str, set[str]] = defaultdict(set)
    positives_by_prompt: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in positive_rows:
        positive_tools_by_prompt[row["user_prompt"]].add(row["tool_call_text"])
        positives_by_prompt[row["user_prompt"]].append(row)

    negative_rows: list[dict[str, str]] = []
    candidate_size_counter: Counter[int] = Counter()

    for prompt, prompt_positive_rows in sorted(positives_by_prompt.items()):
        blocked_tools = positive_tools_by_prompt[prompt]
        candidates = [tool for tool in all_tools if tool not in blocked_tools]
        candidate_size_counter[len(candidates)] += 1

        if len(candidates) < len(prompt_positive_rows):
            raise ValueError(
                "Not enough mismatch candidates for prompt "
                f"{stable_hash(prompt)}: candidates={len(candidates)}, needed={len(prompt_positive_rows)}"
            )

        sampled_tools = rng.sample(candidates, len(prompt_positive_rows))
        for positive_row, negative_tool in zip(prompt_positive_rows, sampled_tools):
            negative_tool_metadata = tool_catalog[negative_tool]
            negative_rows.append(
                {
                    "user_prompt": prompt,
                    "tool_call_text": negative_tool,
                    "label": "1",
                    "prompt_source_file": positive_row["prompt_source_file"],
                    "prompt_source_toolkit": positive_row["prompt_source_toolkit"],
                    "tool_source_file": negative_tool_metadata["tool_source_file"],
                    "tool_source_toolkit": negative_tool_metadata["tool_source_toolkit"],
                    "source_prompt_hash": positive_row["source_prompt_hash"],
                    "normal_tool_call_text": positive_row["tool_call_text"],
                    "mismatch_sampling_type": "random_mismatch_global_tool_excluding_prompt_normal_tools",
                }
            )

    positive_pairs = {(row["user_prompt"], row["tool_call_text"]) for row in positive_rows}
    negative_pairs = {(row["user_prompt"], row["tool_call_text"]) for row in negative_rows}
    invalid_mismatch_overlap = sorted(positive_pairs & negative_pairs)
    if invalid_mismatch_overlap:
        raise AssertionError(f"Mismatch pairs overlap normal pairs: {invalid_mismatch_overlap[:5]}")

    duplicate_mismatch_pairs = len(negative_rows) - len(negative_pairs)
    if duplicate_mismatch_pairs:
        raise AssertionError(f"Duplicate mismatch pairs were generated: {duplicate_mismatch_pairs}")

    summary = {
        "mismatch_rows": len(negative_rows),
        "invalid_mismatch_normal_pair_overlap": len(invalid_mismatch_overlap),
        "duplicate_mismatch_pairs": duplicate_mismatch_pairs,
        "mismatch_candidate_size_min": min(candidate_size_counter) if candidate_size_counter else 0,
        "mismatch_candidate_size_max": max(candidate_size_counter) if candidate_size_counter else 0,
    }
    return negative_rows, summary


def summarize_rows(rows: list[dict[str, str]], prefix: str) -> dict[str, Any]:
    labels = Counter(row["label"] for row in rows)
    prompts = Counter(row["user_prompt"] for row in rows)
    prompt_files = Counter(row["prompt_source_file"] for row in rows)
    prompt_toolkits = Counter(row["prompt_source_toolkit"] for row in rows)
    toolkits = Counter(row["tool_source_toolkit"] for row in rows)
    tools = Counter(row["tool_call_text"] for row in rows)
    triples = Counter((row["user_prompt"], row["tool_call_text"], row["label"]) for row in rows)
    prompt_tool_pairs = Counter((row["user_prompt"], row["tool_call_text"]) for row in rows)

    return {
        f"{prefix}_rows": len(rows),
        f"{prefix}_label_counts": dict(labels),
        f"{prefix}_unique_prompts": len(prompts),
        f"{prefix}_unique_prompt_source_files": len(prompt_files),
        f"{prefix}_unique_prompt_source_toolkits": len(prompt_toolkits),
        f"{prefix}_unique_tool_source_toolkits": len(toolkits),
        f"{prefix}_unique_tools": len(tools),
        f"{prefix}_duplicate_full_rows": sum(1 for count in triples.values() if count > 1),
        f"{prefix}_duplicate_prompt_tool_pairs": sum(1 for count in prompt_tool_pairs.values() if count > 1),
        f"{prefix}_duplicate_prompt_groups": sum(1 for count in prompts.values() if count > 1),
        f"{prefix}_duplicate_prompt_rows": sum(count for count in prompts.values() if count > 1),
        f"{prefix}_max_pairs_per_prompt": max(prompts.values()) if prompts else 0,
        f"{prefix}_top_tools": tools.most_common(20),
        f"{prefix}_top_prompt_source_toolkits": prompt_toolkits.most_common(20),
        f"{prefix}_top_tool_source_toolkits": toolkits.most_common(20),
    }


def main() -> None:
    args = parse_args()
    output_path = timestamped_path(args.output, timestamp=args.timestamp)
    if args.summary_output is None:
        summary_output = output_path.with_suffix(".summary.json")
    else:
        summary_output = timestamped_path(args.summary_output, timestamp=args.timestamp)

    positive_rows, tool_catalog, positive_summary = build_positive_rows(args.source_dir.resolve())
    negative_rows, negative_summary = build_mismatch_rows(
        positive_rows,
        tool_catalog,
        seed=args.seed,
    )

    rows = positive_rows + negative_rows
    random.Random(args.seed).shuffle(rows)

    write_csv(rows, output_path)

    positive_tool_set = {row["tool_call_text"] for row in positive_rows}
    negative_tool_set = {row["tool_call_text"] for row in negative_rows}

    summary = {
        "source_dir": str(args.source_dir.resolve()),
        "output_path": str(output_path),
        "summary_output": str(summary_output),
        "columns": FIELDNAMES,
        "seed": args.seed,
        "label0_definition": "Observed successful ToolBench prompt-tool pair",
        "label1_definition": "Mismatch pair generated by pairing the same prompt with a different observed ToolBench tool while excluding all normal tools for that prompt",
        "prompt_public_processing": "email and URL spans are replaced with [EMAIL] and [URL]",
        "tool_quality_filters": sorted(EXCLUDED_TOOL_PATTERNS),
        **positive_summary,
        **negative_summary,
        **summarize_rows(positive_rows, "label0"),
        **summarize_rows(negative_rows, "label1"),
        **summarize_rows(rows, "combined"),
        "shared_tool_count_between_labels": len(positive_tool_set & negative_tool_set),
        "label1_tools_all_seen_in_label0": negative_tool_set.issubset(positive_tool_set),
    }
    write_json(summary, summary_output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
