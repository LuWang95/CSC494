import argparse
import ast
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOTKA = ROOT / "code" / "residual_neural_ode" / "lotka-volterra"


def fail(message):
    raise AssertionError(message)


def check_python_syntax():
    paths = sorted((ROOT / "code").rglob("*.py")) + [Path(__file__)]
    for path in paths:
        ast.parse(path.read_text(), filename=str(path))
    return f"parsed {len(paths)} Python files"


def function_keywords(path, function_name):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return [argument.arg for argument in node.args.args]
    fail(f"missing function {function_name} in {path}")


def check_big_factor_interface():
    core = LOTKA / "carrying_solver_sweep.py"
    big = LOTKA / "carrying_big_factor_sweep.py"
    parameters = function_keywords(core, "train_one")
    if "regularization_profile" not in parameters:
        fail("train_one does not accept regularization_profile")
    source = big.read_text()
    if "regularization_profile=reg_profile" not in source:
        fail("big-factor sweep does not pass its regularization profile")
    for field in ("regularization_profile", "l2_weight", "ortho_weight"):
        if f'"{field}"' not in core.read_text():
            fail(f"core CSV/training path is missing {field}")
    return "big-factor training interface and regularization fields agree"


def check_manifests_and_interval_metrics():
    experiment_paths = [
        LOTKA / "carrying_solver_sweep.py",
        LOTKA / "carrying_big_factor_sweep.py",
        LOTKA / "carrying_train_interval_sweep.py",
        LOTKA / "carrying_parameter_importance_sweep.py",
    ]
    for path in experiment_paths:
        if "write_run_manifest(" not in path.read_text():
            fail(f"missing run manifest in {path.name}")
    visualizer = (LOTKA / "visualize_carrying_train_interval_sweep.py").read_text()
    interval_source = (LOTKA / "carrying_train_interval_sweep.py").read_text()
    for field in (
        "candidate_ic_extrap_mse_h2p5",
        "candidate_ic_extrap_mse_h5",
        "candidate_ic_extrap_mse_h10",
        "evaluation_status",
        "training_protocol_version",
    ):
        if field not in interval_source:
            fail(f"train-interval sweep omits {field}")
    for field in ("n_covered_ics", "n_partial_ics", "n_uncovered_ics"):
        if field not in visualizer:
            fail(f"train-interval visualizer omits {field}")
    importance_source = (
        LOTKA / "carrying_parameter_importance_sweep.py"
    ).read_text()
    for field in (
        "parameter_rel_error",
        "oracle_gain_candidate_ic_",
        "regularization_profile_names",
    ):
        if field not in importance_source:
            fail(f"parameter-importance sweep omits {field}")
    importance_visualizer = (
        LOTKA / "visualize_carrying_parameter_importance.py"
    ).read_text()
    for field in (
        "fig01_parameter_effect_vs_interval",
        "fig03_interaction_forest",
        "fig05_recovery_heatmaps",
    ):
        if field not in importance_visualizer:
            fail(f"parameter-importance visualizer omits {field}")
    return "run manifests and comparable interval metrics are wired into outputs"


def read_rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def check_unique_rows(path, key_fields, expected_rows, allow_incomplete=False):
    rows = read_rows(path)
    if not rows:
        return f"skipped absent optional result {path.relative_to(ROOT)}"
    keys = [tuple(row.get(field, "") for field in key_fields) for row in rows]
    if len(rows) != expected_rows and not (allow_incomplete and len(rows) < expected_rows):
        fail(f"{path} has {len(rows)} rows; expected {expected_rows}")
    if len(set(keys)) != len(keys):
        fail(f"{path} contains duplicate experiment keys")
    progress = f"{len(rows)}/{expected_rows}" if len(rows) < expected_rows else str(len(rows))
    return f"validated {progress} unique rows in {path.relative_to(ROOT)}"


def check_requirements():
    python_version = (ROOT / ".python-version").read_text().strip()
    if python_version != "3.12":
        fail(f"unexpected Python version pin: {python_version}")
    lines = [
        line.strip()
        for line in (ROOT / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    unpinned = [line for line in lines if "==" not in line]
    if unpinned:
        fail(f"unpinned requirements: {unpinned}")
    return f"validated Python {python_version} and {len(lines)} pinned dependencies"


def parse_args():
    parser = argparse.ArgumentParser(description="Validate CSC494 experiment reproducibility wiring.")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail when the canonical 40-run training-interval result is incomplete.",
    )
    return parser.parse_args()


def main(args):
    checks = [
        check_python_syntax(),
        check_big_factor_interface(),
        check_manifests_and_interval_metrics(),
        check_requirements(),
        check_unique_rows(
            LOTKA / "carrying_solver_sweep_results" / "carrying_solver_sweep_summary.csv",
            ("training_config", "seed"),
            25,
        ),
        check_unique_rows(
            LOTKA / "carrying_big_factor_sweep_results" / "carrying_big_factor_sweep_summary.csv",
            ("training_config", "regularization_profile", "seed"),
            192,
        ),
        check_unique_rows(
            LOTKA / "carrying_train_interval_sweep_results" / "carrying_train_interval_sweep_summary.csv",
            (
                "train_interval",
                "ratio",
                "noise_level",
                "coverage_radius",
                "seed",
                "chunk_size",
                "stage_scale",
                "training_protocol_version",
            ),
            40,
            allow_incomplete=not args.require_complete,
        ),
        check_unique_rows(
            LOTKA
            / "carrying_parameter_importance_sweep_results"
            / "carrying_parameter_importance_sweep_summary.csv",
            (
                "train_interval",
                "regularization_profile",
                "seed",
                "chunk_size",
                "stage_scale",
                "training_protocol_version",
            ),
            160,
            allow_incomplete=True,
        ),
    ]
    for result in checks:
        print(f"[ok] {result}")
    print("[ok] reproducibility checks passed")


if __name__ == "__main__":
    try:
        main(parse_args())
    except (AssertionError, SyntaxError) as error:
        print(f"[failed] {error}", file=sys.stderr)
        raise SystemExit(1)
