#!/usr/bin/env python3
"""
Extracts the changelog section for a given version from CHANGELOG.md.
Usage:
    python extract_changelog.py 0.1.2
Prints the section to stdout.
"""

import re
import sys
from pathlib import Path


def extract_section(version: str, changelog_path: Path) -> str:
    content = changelog_path.read_text()
    # Match the section for the given version (e.g. ## [0.1.2])
    pattern = rf"^## \[{re.escape(version)}\].*?(?=^## |\Z)"
    match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
    if not match:
        print(f"No changelog entry found for version {version}", file=sys.stderr)
        sys.exit(1)
    assert match is not None
    return match.group(0).strip()


def main():
    if len(sys.argv) != 2:
        print("Usage: python extract_changelog.py <version>", file=sys.stderr)
        sys.exit(1)
    version = sys.argv[1]
    changelog_path = Path(__file__).parent.parent.parent / "CHANGELOG.md"
    print(extract_section(version, changelog_path))


if __name__ == "__main__":
    main()
