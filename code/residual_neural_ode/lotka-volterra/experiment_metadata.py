import json
import hashlib
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


SCHEMA_VERSION = 1
TRACKED_PACKAGES = ("jax", "jaxlib", "diffrax", "optax", "numpy")


def package_versions():
    versions = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def source_hashes():
    repo_dir = Path(__file__).resolve().parents[3]
    source_paths = sorted(Path(__file__).resolve().parent.glob("*.py"))
    source_paths.extend([repo_dir / "requirements.txt", repo_dir / "REPRODUCIBILITY.md"])
    hashes = {}
    for path in source_paths:
        if path.exists():
            relative_path = path.relative_to(repo_dir).as_posix()
            hashes[relative_path] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def git_metadata(excluded_path=None):
    repo_dir = Path(__file__).resolve().parents[3]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    status_command = ["git", "status", "--porcelain", "--untracked-files=all", "--", "."]
    if excluded_path is not None:
        try:
            relative_path = Path(excluded_path).resolve().relative_to(repo_dir)
        except ValueError:
            relative_path = None
        if relative_path is not None:
            status_command.append(f":(exclude){relative_path.as_posix()}")
    status = subprocess.run(
        status_command,
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else "unknown",
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def build_run_manifest(experiment, parameters, output_dir=None):
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": experiment,
        "parameters": parameters,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": package_versions(),
        },
        "git": git_metadata(excluded_path=output_dir),
        "source_sha256": source_hashes(),
    }


def write_run_manifest(
    output_dir,
    experiment,
    parameters,
    filename="run_manifest.json",
):
    output_dir = Path(output_dir)
    manifest = build_run_manifest(experiment, parameters, output_dir=output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with path.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest
