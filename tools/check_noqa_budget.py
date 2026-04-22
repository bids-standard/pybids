#!/usr/bin/env python3
"""Fail if '# noqa' directives increase compared to a git ref."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


NOQA_RE = re.compile(r"#\s*(?:ruff:\s*)?noqa\b", re.IGNORECASE)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def tracked_python_files(ref: str) -> list[str]:
    output = git("ls-tree", "-r", "--name-only", ref)
    return [line for line in output.splitlines() if line.endswith(".py")]


def file_content_at_ref(ref: str, path: str) -> str:
    return git("show", f"{ref}:{path}")


def count_noqa_in_text(text: str) -> int:
    return sum(1 for line in text.splitlines() if NOQA_RE.search(line))


def count_noqa_at_ref(ref: str) -> int:
    total = 0
    for path in tracked_python_files(ref):
        try:
            text = file_content_at_ref(ref, path)
        except subprocess.CalledProcessError:
            # File may not exist in one side of comparison.
            continue
        total += count_noqa_in_text(text)
    return total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-ref",
        required=True,
        help="Git ref to compare against (e.g., origin/main).",
    )
    args = parser.parse_args()

    try:
        base_count = count_noqa_at_ref(args.base_ref)
        head_count = count_noqa_at_ref("HEAD")
    except subprocess.CalledProcessError as exc:
        print(f"Failed to run git command: {exc}", file=sys.stderr)
        return 2

    print(f"Base ({args.base_ref}) # noqa count: {base_count}")
    print(f"Head (HEAD) # noqa count: {head_count}")

    if head_count > base_count:
        print(
            f"Error: '# noqa' count increased by {head_count - base_count}. "
            "Please remove added suppressions or justify and adjust policy.",
            file=sys.stderr,
        )
        return 1

    print("OK: '# noqa' count did not increase.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
