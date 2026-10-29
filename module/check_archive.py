"""
Check a built module archive before it is deployed: unpack it into a temp dir and make sure the module can start
from there (VIAM CENTERAL: a tarball without module/live_map.py died on sfhack4 with
"ModuleNotFoundError: No module named 'module.live_map'").

  1. layout: meta.json, its entrypoint (executable), requirements.txt;
  2. every project import in the shipped .py files (module.*, sorter.*, scripts' modules such as
     pick_place_auto) names a file that is in the archive; needs nothing installed, so it always runs;
  3. if this Python has viam-sdk: import module.main and module.sorter_service from the unpacked copy in a clean
     subprocess (-I, cwd = the copy), so the working tree cannot stand in for a missing file.

build.sh runs it on every build; a problem fails the build. Usage: python3 module/check_archive.py [archive]
"""
import ast
import importlib.util
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# Names the shipped code imports from this repo: its packages, and the scripts executor.py puts on sys.path
LOCAL = {"module", "sorter", "scripts"} | {p.stem for p in (REPO / "scripts").glob("*.py")}


def _exists(root: Path, dotted: str) -> bool:
    """A dotted project module is shipped: a .py file, a package dir, or a bare script name under scripts/."""
    parts = dotted.split(".")
    if len(parts) == 1 and parts[0] not in ("module", "sorter", "scripts"):
        return (root / "scripts" / f"{parts[0]}.py").is_file()
    base = root.joinpath(*parts)
    return base.with_suffix(".py").is_file() or base.is_dir()


def missing_imports(root: Path) -> list:
    """(file, line, name) for every project import in the archive's .py files that names a file not shipped."""
    out = []
    for py in sorted(root.rglob("*.py")):
        if py.name.startswith("._"):
            continue  # macOS metadata, reported by check()
        tree = ast.parse(py.read_text(), filename=str(py))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                # "from sorter import executor": executor may be a submodule (file) or a name in sorter/__init__
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names
                                         if (root.joinpath(*node.module.split("."))).is_dir()
                                         and not _defined_in_init(root, node.module, a.name)]
            else:
                continue
            for name in names:
                if name.split(".")[0] in LOCAL and not _exists(root, name):
                    out.append((str(py.relative_to(root)), node.lineno, name))
    return out


def _defined_in_init(root: Path, package: str, name: str) -> bool:
    """`name` is defined at the top of the package's __init__.py (def, class, assignment, import), not a file."""
    init = root.joinpath(*package.split(".")) / "__init__.py"
    if not init.is_file():
        return False
    names = set()
    for n in ast.parse(init.read_text()).body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, ast.Assign):
            names |= {t.id for t in n.targets if isinstance(t, ast.Name)}
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            names |= {(x.asname or x.name).split(".")[0] for x in n.names}
    return name in names


def check(archive: Path) -> list:
    problems = []
    with tempfile.TemporaryDirectory(prefix="module-archive-") as tmp:
        root = Path(tmp)
        with tarfile.open(archive) as tar:
            try:
                tar.extractall(root, filter="data")
            except TypeError:  # Python without extraction filters; the archive is our own build
                tar.extractall(root)
        meta = root / "meta.json"
        if not meta.is_file():
            problems.append("meta.json missing (viam module reload-local rejects the tarball)")
        else:
            entry = root / json.loads(meta.read_text()).get("entrypoint", "")
            if not entry.is_file():
                problems.append(f"entrypoint {entry.name!r} from meta.json is not in the archive")
            elif not entry.stat().st_mode & 0o111:
                problems.append(f"entrypoint {entry.name!r} is not executable")
        if not (root / "requirements.txt").is_file():
            problems.append("requirements.txt missing (run.sh installs from it)")
        junk = [str(p.relative_to(root)) for p in root.rglob("._*")]
        if junk:
            problems.append(f"{len(junk)} macOS ._ metadata files (tar without COPYFILE_DISABLE=1), e.g. {junk[0]}")
        for f, line, name in missing_imports(root):
            problems.append(f"{f}:{line} imports {name!r}, which is not in the archive")
        if importlib.util.find_spec("viam") is None:
            print("check_archive: no viam-sdk in this Python, skipped the import step "
                  "(PYTHON=<venv with viam-sdk> module/build.sh runs it)")
        else:
            code = "import sys; sys.path.insert(0, '.'); import module.main, module.sorter_service; print('ok')"
            r = subprocess.run([sys.executable, "-I", "-c", code], cwd=root, capture_output=True, text=True)
            if r.returncode != 0:
                problems.append("import module.main from the unpacked archive failed:\n"
                                + "\n".join(r.stderr.strip().splitlines()[-4:]))
            else:
                print("check_archive: module.main + module.sorter_service import from the unpacked archive")
    return problems


if __name__ == "__main__":
    path = Path(sys.argv[1] if len(sys.argv) > 1 else REPO / "module" / "dist" / "archive.tar.gz")
    found = check(path)
    for p in found:
        print(f"check_archive: {p}", file=sys.stderr)
    if found:
        sys.exit(1)
    print(f"check_archive: {path.name} OK")
