#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare MIMIC-CXR reports for the RadGraph v3 structuring pipeline.

This script has two stages:

1. Collect MIMIC-CXR report TXT files into a metadata JSONL with fields that
   `radgraph_structure_optimized_v3.py` can read: `Findings` and `Impression`.
2. Optionally call `radgraph_structure_optimized_v3.py` to create the final
   `*_radgraph_structured_v3_full.jsonl` file, which is schema-compatible with
   `rexgradient_radgraph_structured_v3_full.jsonl`.

Examples:

  # Only build the intermediate raw metadata JSONL.
  python scripts/prepare_mimic_radgraph_v3.py ^
    --mimic-root "E:\\程序\\医疗影像项目\\数据\\mimic" ^
    --metadata-output "E:\\程序\\医疗影像项目\\数据\\mimic\\mimic_full_metadata.jsonl"

  # Build metadata, then run the RadGraph v3 script to produce structured JSONL.
  python scripts/prepare_mimic_radgraph_v3.py ^
    --mimic-root "E:\\程序\\医疗影像项目\\数据\\mimic" ^
    --metadata-output "E:\\程序\\医疗影像项目\\数据\\mimic\\mimic_full_metadata.jsonl" ^
    --structured-output "E:\\程序\\医疗影像项目\\数据\\mimic\\mimic_radgraph_structured_v3_full.jsonl" ^
    --run-radgraph ^
    --radgraph-script "E:\\下载\\radgraph_structure_optimized_v3.py" ^
    --local-model-path ".\\models\\radgraph_cache\\manual\\modern-radgraph-xl.tar.gz" ^
    --batch-size 8 ^
    --no-raw
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


DEFAULT_MIMIC_ROOT = r"E:\程序\医疗影像项目\数据\mimic"
DEFAULT_METADATA_NAME = "mimic_full_metadata.jsonl"
DEFAULT_STRUCTURED_NAME = "mimic_radgraph_structured_v3_full.jsonl"


SECTION_ALIASES = {
    "FINDINGS": "findings",
    "FINDING": "findings",
    "CHEST FINDINGS": "findings",
    "RADIOGRAPHIC FINDINGS": "findings",
    "IMPRESSION": "impression",
    "IMPRESSIONS": "impression",
    "CONCLUSION": "impression",
    "CONCLUSIONS": "impression",
}

HEADER_RE = re.compile(r"^\s*([A-Z][A-Z0-9 /,()\-]+?)\s*:\s*(.*)$")
REPORT_PATH_RE = re.compile(r"(?:^|/)files/p\d+/p(?P<subject_id>\d+)/s(?P<study_id>\d+)\.txt$")


@dataclass(frozen=True)
class ReportItem:
    """One raw MIMIC-CXR report."""

    report_path: str
    text: str


def _clean_text(text: str) -> str:
    """Normalize whitespace while preserving sentence content."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.splitlines()]
    # Collapse repeated blank lines, then convert remaining newlines to spaces.
    out: list[str] = []
    blank_seen = False
    for line in lines:
        if not line:
            if not blank_seen:
                out.append("")
            blank_seen = True
            continue
        out.append(line)
        blank_seen = False
    return re.sub(r"\s+", " ", "\n".join(out)).strip()


def _normalize_header(label: str) -> str:
    label = re.sub(r"\s+", " ", label.strip().upper())
    return label.rstrip(".")


def split_report_sections(text: str) -> dict[str, str]:
    """Extract major sections from a MIMIC-CXR report TXT.

    MIMIC reports usually contain all-caps headers such as `FINDINGS:` and
    `IMPRESSION:`. The parser is line-based and keeps inline content after the
    colon, e.g. `IMPRESSION: No acute disease.`
    """
    sections: dict[str, list[str]] = {
        "findings": [],
        "impression": [],
    }
    current: str | None = None

    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        line = raw_line.rstrip()
        match = HEADER_RE.match(line)
        if match:
            label = _normalize_header(match.group(1))
            canonical = SECTION_ALIASES.get(label)
            current = canonical
            if canonical and match.group(2).strip():
                sections[canonical].append(match.group(2).strip())
            continue
        if current:
            sections[current].append(line.strip())

    return {key: _clean_text("\n".join(value)) for key, value in sections.items()}


def ids_from_report_path(report_path: str) -> tuple[str, str]:
    """Return `(subject_id, study_id)` parsed from a standard MIMIC path."""
    normalized = report_path.replace("\\", "/")
    match = REPORT_PATH_RE.search(normalized)
    if not match:
        return "", ""
    return match.group("subject_id"), match.group("study_id")


def iter_reports_from_files(files_dir: Path) -> Iterator[ReportItem]:
    """Yield reports from an extracted MIMIC `files/` directory."""
    for path in sorted(files_dir.glob("p*/p*/s*.txt")):
        yield ReportItem(
            report_path=path.as_posix(),
            text=path.read_text(encoding="utf-8", errors="replace"),
        )


def iter_reports_from_zip(zip_path: Path) -> Iterator[ReportItem]:
    """Yield reports directly from a `mimic-cxr-reports.zip` archive."""
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(
            name for name in zf.namelist()
            if name.startswith("files/") and name.endswith(".txt")
        )
        for name in names:
            with zf.open(name) as fh:
                text = fh.read().decode("utf-8", errors="replace")
            yield ReportItem(report_path=name, text=text)


def find_report_source(mimic_root: Path, source: str) -> tuple[str, Path]:
    """Resolve whether to read extracted files or the zip archive."""
    if mimic_root.is_file() and mimic_root.suffix.lower() == ".zip":
        return "zip", mimic_root

    files_dir = mimic_root / "files"
    zip_path = mimic_root / "mimic-cxr-reports.zip"

    if source == "files":
        if not files_dir.exists():
            raise FileNotFoundError(f"Extracted files directory not found: {files_dir}")
        return "files", files_dir
    if source == "zip":
        if not zip_path.exists():
            raise FileNotFoundError(f"Zip archive not found: {zip_path}")
        return "zip", zip_path

    if files_dir.exists():
        return "files", files_dir
    if zip_path.exists():
        return "zip", zip_path
    raise FileNotFoundError(
        f"Could not find MIMIC reports under {mimic_root}. "
        "Expected either files/ or mimic-cxr-reports.zip."
    )


def iter_reports(mimic_root: Path, source: str) -> Iterator[ReportItem]:
    source_kind, source_path = find_report_source(mimic_root, source)
    if source_kind == "files":
        yield from iter_reports_from_files(source_path)
    else:
        yield from iter_reports_from_zip(source_path)


def build_metadata_jsonl(args: argparse.Namespace) -> int:
    """Write raw MIMIC report metadata JSONL for RadGraph processing."""
    mimic_root = Path(args.mimic_root)
    output_path = Path(args.metadata_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_seen = 0
    n_missing_findings = 0
    n_missing_impression = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for item in iter_reports(mimic_root, args.source):
            if args.max_records >= 0 and n_seen >= args.max_records:
                break
            n_seen += 1

            sections = split_report_sections(item.text)
            findings = sections.get("findings", "")
            impression = sections.get("impression", "")
            if not findings:
                n_missing_findings += 1
            if not impression:
                n_missing_impression += 1
            if args.require_findings and not findings:
                continue
            if args.require_impression and not impression:
                continue
            if not findings and not impression:
                continue

            subject_id, study_id = ids_from_report_path(item.report_path)
            record = {
                "id": f"mimic-cxr-{study_id}" if study_id else f"mimic-cxr-{n_seen}",
                "source_dataset": "mimic-cxr",
                "subject_id": subject_id,
                "study_id": study_id,
                "report_path": item.report_path,
                "Findings": findings,
                "Impression": impression,
            }
            if args.include_raw_report:
                record["raw_report"] = item.text

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[metadata] source={mimic_root}")
    print(f"[metadata] output={output_path}")
    print(f"[metadata] seen={n_seen} written={n_written}")
    print(f"[metadata] missing_findings={n_missing_findings} missing_impression={n_missing_impression}")
    return n_written


def run_radgraph_script(args: argparse.Namespace) -> None:
    """Call `radgraph_structure_optimized_v3.py` on the metadata JSONL."""
    script_path = Path(args.radgraph_script)
    if not script_path.exists():
        raise FileNotFoundError(f"RadGraph structure script not found: {script_path}")

    structured_output = Path(args.structured_output)
    structured_output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(script_path),
        "--input-file",
        str(args.metadata_output),
        "--output-file",
        str(structured_output),
        "--model-type",
        args.model_type,
        "--batch-size",
        str(args.batch_size),
        "--max-records",
        str(args.max_records),
        "--cuda",
        args.cuda,
        "--cache-dir",
        args.cache_dir,
        "--findings-keys",
        "Findings",
        "findings",
        "FINDINGS",
        "--impression-keys",
        "Impression",
        "impression",
        "IMPRESSION",
    ]

    optional_flags = [
        ("--local-model-path", args.local_model_path),
        ("--hf-endpoint", args.hf_endpoint),
    ]
    for flag, value in optional_flags:
        if value:
            cmd.extend([flag, value])

    bool_flags = [
        ("--local-files-only", args.local_files_only),
        ("--no-raw", args.no_raw),
        ("--no-impression-graph", args.no_impression_graph),
        ("--no-fail-soft", args.no_fail_soft),
        ("--no-normalize-text", args.no_normalize_text),
        ("--show-suggestive-targets-in-compact", args.show_suggestive_targets_in_compact),
    ]
    for flag, enabled in bool_flags:
        if enabled:
            cmd.append(flag)

    if not args.merge_adjacent_observations:
        cmd.append("--no-merge-adjacent-observations")
    cmd.extend(["--adjacent-merge-max-gap", str(args.adjacent_merge_max_gap)])
    cmd.extend(["--preview-n", str(args.preview_n)])
    cmd.extend(["--preview-max-items", str(args.preview_max_items)])

    print("[radgraph] running:")
    print(" ".join(f'"{part}"' if " " in part else part for part in cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect MIMIC-CXR reports and optionally run RadGraph v3 to create "
            "rexgradient_radgraph_structured_v3_full.jsonl-compatible data."
        )
    )
    parser.add_argument("--mimic-root", default=DEFAULT_MIMIC_ROOT,
                        help="MIMIC directory containing files/ and/or mimic-cxr-reports.zip.")
    parser.add_argument("--source", choices=["auto", "files", "zip"], default="auto",
                        help="Read extracted files, zip archive, or auto-detect. Default: auto.")
    parser.add_argument("--metadata-output", default="",
                        help="Intermediate metadata JSONL path. Default: <mimic-root>/mimic_full_metadata.jsonl.")
    parser.add_argument("--structured-output", default="",
                        help="Final structured JSONL path. Default: <mimic-root>/mimic_radgraph_structured_v3_full.jsonl.")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Maximum reports to collect/process. Use -1 for full dataset.")
    parser.add_argument("--require-findings", action="store_true",
                        help="Skip reports where the FINDINGS section could not be parsed.")
    parser.add_argument("--require-impression", action="store_true",
                        help="Skip reports where the IMPRESSION section could not be parsed.")
    parser.add_argument("--include-raw-report", action="store_true",
                        help="Include full raw report text in metadata JSONL. Off by default to keep files smaller.")

    parser.add_argument("--run-radgraph", action="store_true",
                        help="After metadata collection, run radgraph_structure_optimized_v3.py.")
    parser.add_argument("--radgraph-script", default=r"E:\下载\radgraph_structure_optimized_v3.py",
                        help="Path to radgraph_structure_optimized_v3.py.")
    parser.add_argument("--model-type", default="modern-radgraph-xl")
    parser.add_argument("--local-model-path", default="")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cuda", default="auto")
    parser.add_argument("--cache-dir", default="./models/radgraph_cache")
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-raw", action="store_true")
    parser.add_argument("--no-impression-graph", action="store_true")
    parser.add_argument("--no-fail-soft", action="store_true")
    parser.add_argument("--no-normalize-text", action="store_true")
    parser.add_argument("--show-suggestive-targets-in-compact", action="store_true")
    parser.add_argument("--no-merge-adjacent-observations", dest="merge_adjacent_observations",
                        action="store_false")
    parser.set_defaults(merge_adjacent_observations=True)
    parser.add_argument("--adjacent-merge-max-gap", type=int, default=1)
    parser.add_argument("--preview-n", type=int, default=3)
    parser.add_argument("--preview-max-items", type=int, default=8)

    args = parser.parse_args()
    mimic_root = Path(args.mimic_root)
    if not args.metadata_output:
        args.metadata_output = str(mimic_root / DEFAULT_METADATA_NAME)
    if not args.structured_output:
        args.structured_output = str(mimic_root / DEFAULT_STRUCTURED_NAME)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_records < -1:
        raise ValueError("--max-records must be -1 or a non-negative integer")
    return args


def main() -> None:
    args = parse_args()
    n_written = build_metadata_jsonl(args)
    if args.run_radgraph:
        if n_written == 0:
            raise RuntimeError("No metadata records were written; refusing to run RadGraph.")
        run_radgraph_script(args)
    else:
        print("[done] Metadata JSONL created. Re-run with --run-radgraph to create structured JSONL.")


if __name__ == "__main__":
    main()
