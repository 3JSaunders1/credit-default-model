"""
tools/export_codebase.py
Prints the project's folder structure and saves a snapshot of all
project files (code, SQL, docs) into tools/snapshots/.

Usage:
  python tools/export_codebase.py          # tree + snapshot (asks first)
  python tools/export_codebase.py --tree   # tree only
"""
import sys
from datetime import datetime
from pathlib import Path

IGNORED_DIRS = {"venv", ".venv", "__pycache__", ".git", ".pytest_cache",
                ".ipynb_checkpoints", "snapshots", "site-packages"}
INCLUDED_SUFFIXES = {".py", ".sql", ".md", ".txt", ".toml", ".yaml", ".yml", ".ini", ".csv"}
INCLUDED_NAMES = {"Makefile", ".gitignore"}
SKIP_DATA_DIRS = {"raw", "interim", "processed"}   # list them, don't copy data


def find_project_root(start: Path) -> Path:
    """Walk upward until we find the folder containing src/ and data/."""
    for parent in [start, *start.parents]:
        if (parent / "src").exists() and (parent / "data").exists():
            return parent
    return start.parent


def is_ignored(path: Path, root: Path) -> bool:
    return any(part in IGNORED_DIRS for part in path.relative_to(root).parts)


def build_tree(folder: Path, root: Path, prefix: str = "") -> list[str]:
    """Return the folder tree as lines, like the `tree` command."""
    lines = []
    items = sorted(
        (p for p in folder.iterdir() if not is_ignored(p, root)),
        key=lambda p: (p.is_file(), p.name.lower()),   # folders first
    )
    for i, item in enumerate(items):
        last = i == len(items) - 1
        branch = "└── " if last else "├── "
        lines.append(f"{prefix}{branch}{item.name}{'/' if item.is_dir() else ''}")
        if item.is_dir():
            extension = "    " if last else "│   "
            lines.extend(build_tree(item, root, prefix + extension))
    return lines


def collect_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if not path.is_file() or is_ignored(path, root):
            continue
        parts = path.relative_to(root).parts
        if parts[0] == "data" and len(parts) > 1 and parts[1] in SKIP_DATA_DIRS:
            continue   # never copy datasets into the snapshot
        if path.suffix in INCLUDED_SUFFIXES or path.name in INCLUDED_NAMES:
            files.append(path)
    return sorted(files)


def build_snapshot(root: Path, tree_lines: list[str], files: list[Path]) -> str:
    out = [
        "# CODEBASE SNAPSHOT",
        f"# Root: {root}",
        f"# Generated: {datetime.now().isoformat()}",
        "",
        "# PROJECT STRUCTURE",
        f"{root.name}/",
        *tree_lines,
        "",
        "=" * 100,
    ]
    for f in files:
        out += ["", "#" * 100, f"# FILE: {f.relative_to(root)}", "#" * 100, ""]
        try:
            out.append(f.read_text(encoding="utf-8"))
        except Exception as e:
            out.append(f"# ERROR READING FILE: {e}")
    return "\n".join(out)


def main():
    root = find_project_root(Path(__file__).resolve().parent)
    tree_lines = build_tree(root, root)

    print(f"\n{root.name}/")
    print("\n".join(tree_lines))

    if "--tree" in sys.argv:
        return

    answer = input("\nSave a codebase snapshot? (y/n): ").strip().lower()
    if answer != "y":
        print("No snapshot created.")
        return

    files = collect_files(root)
    snapshot = build_snapshot(root, tree_lines, files)

    out_dir = root / "tools" / "snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"codebase_snapshot_{datetime.now():%Y%m%d_%H%M%S}.txt"
    out_file.write_text(snapshot, encoding="utf-8")
    print(f"\nSaved {len(files)} files to:\n{out_file}")


if __name__ == "__main__":
    main()