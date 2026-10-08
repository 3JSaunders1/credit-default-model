"""
One-time refactor: add structured logging to every pipeline module.

Usage: python tools/add_logging.py <package_dir> <import_line>
"""
import re
import sys
from pathlib import Path

SKIP = {"__init__.py", "logging_utils.py", "config.py"}
MAIN_CALL = re.compile(r'if __name__ == "__main__":\n(\s+)main\(\)')
SAVED_PRINT = re.compile(r'^(\s*)print\((f?)"(?:\\n)?(Saved [^"]*)"\)\s*$', re.M)
TOP_LEVEL_DEF = re.compile(r"^(def |class )", re.M)


def refactor(path: Path, import_line: str) -> list[str]:
    text = path.read_text()
    notes = []
    if "get_logger" in text:
        return ["already has logging, skipped"]

    # 1. Logger import + module logger, before the first top-level def/class
    m = TOP_LEVEL_DEF.search(text)
    if not m:
        return ["no top-level def/class found, skipped"]
    block = f"{import_line}\n\nlog = get_logger(__name__)\n\n\n"
    text = text[:m.start()] + block + text[m.start():]

    # 2. Wrap main() so start, duration, and failures are logged
    text, n_main = MAIN_CALL.subn(r'if __name__ == "__main__":\n\1run_main(main)', text)
    if n_main == 0:
        notes.append("no `main()` entry point found; check manually")

    # 3. "Saved ..." prints become log.info
    text, n_saved = SAVED_PRINT.subn(r'\1log.info(\2"\3")', text)
    notes.append(f"{n_saved} 'Saved' print(s) converted")

    path.write_text(text)
    return notes


def main():
    pkg, import_line = Path(sys.argv[1]), sys.argv[2]
    for path in sorted(pkg.glob("*.py")):
        if path.name in SKIP:
            continue
        print(f"{path.name}: {'; '.join(refactor(path, import_line))}")


if __name__ == "__main__":
    main()
