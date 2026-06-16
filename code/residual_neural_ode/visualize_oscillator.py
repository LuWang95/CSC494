import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COMPONENTS = ("x", "v")


def as_array(value):
    return np.asarray(value)


def get_time_grid(results, trajectory):
    if "t_obs" in results:
        return as_array(results["t_obs"])

    dt_obs = float(results["h_model"]) * int(results["ratio"])
    return np.arange(trajectory.shape[0]) * dt_obs


def metric(results, key):
    value = results.get(key)
    if value is None:
        return "N/A"
    return f"{float(value):.6e}"


def experiment_title(results):
    return results.get("experiment_type", "Duffing oscillator experiment")


def plot_trajectories(results, save_dir=None):
    true_train = as_array(results["true_traj"])
    pred_train = as_array(results["pred_traj"])
    true_val = as_array(results.get("true_validation_traj", true_train))
    pred_val = as_array(results.get("pred_validation_traj", pred_train))
    t_obs = get_time_grid(results, true_train)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"{experiment_title(results)}: trajectory rollout vs reference",
        fontsize=14,
    )
    for i, label in enumerate(COMPONENTS):
        train_ax = axes[0, i]
        train_ax.plot(t_obs, true_train[:, i], "--", alpha=0.7, label="true")
        train_ax.plot(t_obs, pred_train[:, i], alpha=0.9, label="predicted")
        train_ax.set_xlabel("t")
        train_ax.set_ylabel(label)
        train_ax.set_title(f"Training initial condition: {label}(t)")
        train_ax.legend()
        train_ax.grid(True)

        val_ax = axes[1, i]
        val_ax.plot(t_obs, true_val[:, i], "--", alpha=0.7, label="true")
        val_ax.plot(t_obs, pred_val[:, i], alpha=0.9, label="predicted")
        val_ax.set_xlabel("t")
        val_ax.set_ylabel(label)
        val_ax.set_title(f"Validation initial condition: {label}(t)")
        val_ax.legend()
        val_ax.grid(True)

    fig.tight_layout()
    if save_dir is not None:
        fig.savefig(save_dir / "trajectories.png", dpi=200)


def vector_field_grid(results):
    states = as_array(results["states"])
    x_vals = np.unique(states[:, 0])
    v_vals = np.unique(states[:, 1])
    x_grid, v_grid = np.meshgrid(x_vals, v_vals)
    return x_grid, v_grid, len(x_vals), len(v_vals)


def plot_quiver(ax, x_grid, v_grid, vf, nx, ny, title):
    vf = as_array(vf)
    u = vf[:, 0].reshape(ny, nx)
    w = vf[:, 1].reshape(ny, nx)
    ax.quiver(x_grid, v_grid, u, w, angles="xy")
    ax.set_xlabel("x")
    ax.set_ylabel("v")
    ax.set_title(title)


def plot_vector_fields(results, save_dir=None):
    x_grid, v_grid, nx, ny = vector_field_grid(results)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_quiver(
        axes[0],
        x_grid,
        v_grid,
        results["true_vf"],
        nx,
        ny,
        "Ground-truth vector field\nDuffing oscillator dynamics",
    )
    plot_quiver(
        axes[1],
        x_grid,
        v_grid,
        results["physics_vf"],
        nx,
        ny,
        "Physics-prior vector field\nknown/assumed terms only",
    )
    plot_quiver(
        axes[2],
        x_grid,
        v_grid,
        results["model_vf"],
        nx,
        ny,
        "Learned full vector field\nphysics prior + NN residual",
    )
    fig.tight_layout()
    if save_dir is not None:
        fig.savefig(save_dir / "vector_fields.png", dpi=200)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_quiver(
        axes[0],
        x_grid,
        v_grid,
        results["true_residual"],
        nx,
        ny,
        "Ground-truth missing dynamics\ntrue field - physics prior",
    )
    plot_quiver(
        axes[1],
        x_grid,
        v_grid,
        results["learned_residual"],
        nx,
        ny,
        "Learned missing dynamics\nNN residual field",
    )
    fig.tight_layout()
    if save_dir is not None:
        fig.savefig(save_dir / "residual_fields.png", dpi=200)


def print_trajectory_check(title, t_obs, pred, true):
    print(f"\n{title}: first 5 observed points")
    print("-" * 80)
    print(
        f"{'t':>8} | "
        f"{'pred x':>12} {'true x':>12} {'abs err':>10} | "
        f"{'pred v':>12} {'true v':>12} {'abs err':>10}"
    )
    print("-" * 80)

    for i in range(min(5, len(t_obs))):
        x_err = abs(float(pred[i, 0] - true[i, 0]))
        v_err = abs(float(pred[i, 1] - true[i, 1]))
        print(
            f"{float(t_obs[i]):8.3f} | "
            f"{float(pred[i, 0]):12.6f} "
            f"{float(true[i, 0]):12.6f} "
            f"{x_err:10.4e} | "
            f"{float(pred[i, 1]):12.6f} "
            f"{float(true[i, 1]):12.6f} "
            f"{v_err:10.4e}"
        )


def print_summary(results):
    true_train = as_array(results["true_traj"])
    pred_train = as_array(results["pred_traj"])
    true_val = as_array(results.get("true_validation_traj", true_train))
    pred_val = as_array(results.get("pred_validation_traj", pred_train))
    t_obs = get_time_grid(results, true_train)

    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"experiment type        : {results.get('experiment_type', 'unknown')}")
    print(f"ratio                  : {results.get('ratio', 'N/A')}")
    print(f"h_model                : {float(results['h_model']):.6f}")
    print(f"best loss              : {metric(results, 'best_loss')}")
    print(f"validation loss        : {metric(results, 'validation_loss')}")
    print(f"learned physics params : {results.get('learned_f_physics', 'N/A')}")
    print("=" * 80)

    print_trajectory_check("Train trajectory check", t_obs, pred_train, true_train)
    print_trajectory_check("Validation trajectory check", t_obs, pred_val, true_val)

    print("\nGlobal metrics on vector-field grid")
    print("-" * 80)
    print(f"train batch MSE             : {metric(results, 'train_batch_mse')}")
    print(f"train batch rel error       : {metric(results, 'train_batch_rel_error')}")
    print(f"validation batch MSE        : {metric(results, 'validation_batch_mse')}")
    print(
        "validation batch rel error  : "
        f"{metric(results, 'validation_batch_rel_error')}"
    )
    print(f"train sample MSE            : {metric(results, 'sample_traj_mse')}")
    print(f"train sample rel error      : {metric(results, 'sample_traj_rel_error')}")
    print(f"validation sample MSE       : {metric(results, 'validation_sample_mse')}")
    print(
        "validation sample rel error : "
        f"{metric(results, 'validation_sample_rel_error')}"
    )
    print(f"residual MSE                : {metric(results, 'residual_mse')}")
    print(f"residual relative error     : {metric(results, 'residual_rel_error')}")
    print(f"model vector field MSE      : {metric(results, 'model_vf_mse')}")
    print(f"model vector field rel error: {metric(results, 'model_vf_rel_error')}")
    print("=" * 80)


def visualize_results(results, save_dir=None, show=True):
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    print_summary(results)
    plot_trajectories(results, save_dir)
    plot_vector_fields(results, save_dir)
    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize saved Duffing oscillator residual Neural ODE results."
    )
    parser.add_argument(
        "result_path",
        nargs="?",
        default="oscillator_wrong_physics_trainable.pkl",
        help="Path to a saved result pickle.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional directory to save figures instead of only showing them.",
    )
    args = parser.parse_args()

    result_path = Path(args.result_path)
    with result_path.open("rb") as f:
        results = pickle.load(f)

    visualize_results(results, args.save_dir)


if __name__ == "__main__":
    main()
