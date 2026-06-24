import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SPECIES = ("prey", "predator")


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
    return results.get("experiment_type", "Lotka-Volterra experiment")


def experiment_metadata(results):
    solver_type = results.get(
        "solver_type",
        results.get("solver_method", results.get("method", "rk4")),
    )
    step_size = results.get(
        "integration_step_size",
        results.get("solver_step_size", results.get("h_model")),
    )
    ratio = results.get("ratio")
    noise_level = results.get("noise_level", results.get("train_noise_level"))

    metadata = [f"solver: {solver_type}"]
    if step_size is not None:
        metadata.append(f"step size: {float(step_size):.6g}")
    if ratio is not None:
        metadata.append(f"ratio: {ratio}")
    metadata.append(
        f"noise level: {float(noise_level):.6g}"
        if noise_level is not None
        else "noise level: N/A"
    )
    return " | ".join(metadata)


def add_experiment_metadata(fig, results):
    fig.text(
        0.5,
        0.01,
        experiment_metadata(results),
        ha="center",
        va="bottom",
        fontsize=10,
    )


def plot_trajectories(results, save_dir=None):
    true_train = as_array(results["true_traj"])
    noisy_train = as_array(results.get("noisy_train_traj", true_train))
    pred_train = as_array(results["pred_traj"])
    true_train_plot = as_array(results.get("true_extrapolate_traj", true_train))
    pred_train_plot = as_array(results.get("pred_extrapolate_traj", pred_train))
    true_val = as_array(results.get("true_validation_traj", true_train))
    pred_val = as_array(results.get("pred_validation_traj", pred_train))
    t_obs = get_time_grid(results, true_train)
    t_plot = as_array(results.get("t_extrapolate", t_obs))
    extrapolation_start = float(results.get("extrapolation_start", t_obs[-1]))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"{experiment_title(results)}: trajectory rollout vs reference",
        fontsize=14,
    )
    for i, label in enumerate(SPECIES):
        train_ax = axes[0, i]
        train_ax.plot(
            t_plot,
            true_train_plot[:, i],
            "k--",
            alpha=0.8,
            label="clean true",
        )
        train_ax.scatter(
            t_obs,
            noisy_train[:, i],
            color="tab:orange",
            s=10,
            alpha=0.55,
            label="noisy observation",
        )
        train_ax.plot(
            t_plot,
            pred_train_plot[:, i],
            color="tab:blue",
            alpha=0.9,
            label="predicted",
        )
        if t_plot[-1] > t_obs[-1]:
            train_ax.axvline(
                extrapolation_start,
                color="tab:red",
                linestyle=":",
                linewidth=1.5,
                label="extrapolation starts",
            )
            train_ax.axvspan(
                extrapolation_start,
                float(t_plot[-1]),
                color="tab:red",
                alpha=0.06,
            )
        train_ax.set_xlabel("t")
        train_ax.set_ylabel(label)
        train_ax.set_title(f"Training initial condition: {label}(t)")
        train_ax.legend()
        train_ax.grid(True)

        val_ax = axes[1, i]
        val_ax.plot(t_obs, true_val[:, i], "k--", alpha=0.8, label="clean true")
        val_ax.plot(
            t_obs,
            pred_val[:, i],
            color="tab:blue",
            alpha=0.9,
            label="predicted",
        )
        val_ax.set_xlabel("t")
        val_ax.set_ylabel(label)
        val_ax.set_title(f"Validation initial condition: {label}(t)")
        val_ax.legend()
        val_ax.grid(True)

    add_experiment_metadata(fig, results)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    if save_dir is not None:
        fig.savefig(save_dir / "trajectories.png", dpi=200)


def vector_field_grid(results):
    states = as_array(results["states"])
    prey_vals = np.unique(states[:, 0])
    pred_vals = np.unique(states[:, 1])
    prey_grid, pred_grid = np.meshgrid(prey_vals, pred_vals)
    return prey_grid, pred_grid, len(prey_vals), len(pred_vals)


def plot_quiver(ax, prey_grid, pred_grid, vf, nx, ny, title):
    vf = as_array(vf)
    u = vf[:, 0].reshape(ny, nx)
    v = vf[:, 1].reshape(ny, nx)
    ax.quiver(prey_grid, pred_grid, u, v, angles="xy")
    ax.set_xlabel("prey")
    ax.set_ylabel("predator")
    ax.set_title(title)


def vector_components(vf, nx, ny):
    vf = as_array(vf)
    return vf[:, 0].reshape(ny, nx), vf[:, 1].reshape(ny, nx)


def normalized_vector_components(vf, nx, ny):
    u, v = vector_components(vf, nx, ny)
    norm = np.sqrt(u**2 + v**2)
    norm = np.where(norm > 0, norm, 1.0)
    return u / norm, v / norm

def plot_vector_field_raw_overlay(results, prey_grid, pred_grid, nx, ny, save_dir=None):
    true_vf = as_array(results["true_vf"])
    model_vf = as_array(results["model_vf"])

    true_u, true_v = vector_components(true_vf, nx, ny)
    model_u, model_v = vector_components(model_vf, nx, ny)

    error_norm = np.linalg.norm(model_vf - true_vf, axis=1).reshape(ny, nx)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.suptitle(
        f"{experiment_title(results)}: raw true vs model vector field",
        fontsize=14,
    )

    error_map = ax.contourf(
        prey_grid,
        pred_grid,
        error_norm,
        levels=20,
        cmap="magma",
        alpha=0.45,
    )
    fig.colorbar(error_map, ax=ax, label="||model rhs - true rhs||")

    ax.quiver(
        prey_grid,
        pred_grid,
        model_u,
        model_v,
        color="tab:blue",
        angles="xy",
        alpha=0.65,
        label="model field",
    )
    ax.quiver(
        prey_grid,
        pred_grid,
        true_u,
        true_v,
        color="black",
        angles="xy",
        alpha=0.9,
        label="true field",
    )

    ax.set_xlabel("prey")
    ax.set_ylabel("predator")
    ax.set_title("Raw vector fields with RHS error magnitude")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)

    add_experiment_metadata(fig, results)
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))

    if save_dir is not None:
        fig.savefig(save_dir / "vector_field_overlay_raw.png", dpi=200)

def plot_vector_field_overlay(results, prey_grid, pred_grid, nx, ny, save_dir=None):
    true_vf = as_array(results["true_vf"])
    model_vf = as_array(results["model_vf"])
    true_u, true_v = normalized_vector_components(true_vf, nx, ny)
    model_u, model_v = normalized_vector_components(model_vf, nx, ny)
    error_norm = np.linalg.norm(model_vf - true_vf, axis=1).reshape(ny, nx)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.suptitle(
        f"{experiment_title(results)}: true vs model vector field",
        fontsize=14,
    )
    error_map = ax.contourf(
        prey_grid,
        pred_grid,
        error_norm,
        levels=20,
        cmap="magma",
        alpha=0.45,
    )
    fig.colorbar(error_map, ax=ax, label="||model rhs - true rhs||")
    ax.quiver(
        prey_grid,
        pred_grid,
        model_u,
        model_v,
        color="tab:blue",
        angles="xy",
        scale=25,
        width=0.006,
        alpha=0.65,
        label="model direction",
    )
    ax.quiver(
        prey_grid,
        pred_grid,
        true_u,
        true_v,
        color="black",
        angles="xy",
        scale=25,
        width=0.0025,
        alpha=0.95,
        label="true direction",
    )
    ax.set_xlabel("prey")
    ax.set_ylabel("predator")
    ax.set_title("Overlayed normalized directions with RHS error magnitude")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    add_experiment_metadata(fig, results)
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    if save_dir is not None:
        fig.savefig(save_dir / "vector_field_overlay.png", dpi=200)


def plot_vector_field_streamplot(results, prey_grid, pred_grid, nx, ny, save_dir=None):
    true_vf = as_array(results["true_vf"])
    model_vf = as_array(results["model_vf"])
    true_u, true_v = vector_components(true_vf, nx, ny)
    model_u, model_v = vector_components(model_vf, nx, ny)
    error_norm = np.linalg.norm(model_vf - true_vf, axis=1).reshape(ny, nx)
    prey_axis = prey_grid[0, :]
    pred_axis = pred_grid[:, 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    fig.suptitle(
        f"{experiment_title(results)}: streamlines over RHS error",
        fontsize=14,
    )

    error_maps = []
    for ax, u, v, title, color in (
        (axes[0], true_u, true_v, "True vector field streamlines", "black"),
        (axes[1], model_u, model_v, "Model vector field streamlines", "tab:blue"),
    ):
        error_map = ax.contourf(
            prey_grid,
            pred_grid,
            error_norm,
            levels=20,
            cmap="magma",
            alpha=0.45,
        )
        error_maps.append(error_map)
        ax.streamplot(
            prey_axis,
            pred_axis,
            u,
            v,
            color=color,
            density=1.2,
            linewidth=1.2,
            arrowsize=1.2,
        )
        ax.set_xlabel("prey")
        ax.set_ylabel("predator")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)

    cbar_ax = fig.add_axes([0.92, 0.18, 0.018, 0.62])
    fig.colorbar(
        error_maps[-1],
        cax=cbar_ax,
        label="||model rhs - true rhs||",
    )
    add_experiment_metadata(fig, results)
    fig.subplots_adjust(left=0.07, right=0.86, bottom=0.14, top=0.84, wspace=0.16)
    if save_dir is not None:
        fig.savefig(save_dir / "vector_field_streamplot.png", dpi=200)


def plot_vector_fields(results, save_dir=None):
    prey_grid, pred_grid, nx, ny = vector_field_grid(results)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        f"{experiment_title(results)}: vector fields",
        fontsize=14,
    )
    plot_quiver(
        axes[0],
        prey_grid,
        pred_grid,
        results["true_vf"],
        nx,
        ny,
        "Ground-truth vector field\nLotka-Volterra dynamics",
    )
    plot_quiver(
        axes[1],
        prey_grid,
        pred_grid,
        results["physics_vf"],
        nx,
        ny,
        "Physics-prior vector field\nknown/assumed terms only",
    )
    plot_quiver(
        axes[2],
        prey_grid,
        pred_grid,
        results["model_vf"],
        nx,
        ny,
        "Learned full vector field\nphysics prior + NN residual",
    )
    add_experiment_metadata(fig, results)
    fig.tight_layout(rect=(0, 0.08, 1, 0.88))
    if save_dir is not None:
        fig.savefig(save_dir / "vector_fields.png", dpi=200)

    plot_vector_field_overlay(results, prey_grid, pred_grid, nx, ny, save_dir)
    plot_vector_field_streamplot(results, prey_grid, pred_grid, nx, ny, save_dir)
    plot_vector_field_raw_overlay(results, prey_grid, pred_grid, nx, ny, save_dir)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(
        f"{experiment_title(results)}: residual fields",
        fontsize=14,
    )
    plot_quiver(
        axes[0],
        prey_grid,
        pred_grid,
        results["true_residual"],
        nx,
        ny,
        "Ground-truth missing dynamics\ntrue field - physics prior",
    )
    plot_quiver(
        axes[1],
        prey_grid,
        pred_grid,
        results["learned_residual"],
        nx,
        ny,
        "Learned missing dynamics\nNN residual field",
    )
    add_experiment_metadata(fig, results)
    fig.tight_layout(rect=(0, 0.08, 1, 0.88))
    if save_dir is not None:
        fig.savefig(save_dir / "residual_fields.png", dpi=200)


def print_trajectory_check(title, t_obs, pred, true):
    print(f"\n{title}: first 5 observed points")
    print("-" * 80)
    print(
        f"{'t':>8} | "
        f"{'pred prey':>12} {'true prey':>12} {'abs err':>10} | "
        f"{'pred pred':>12} {'true pred':>12} {'abs err':>10}"
    )
    print("-" * 80)

    for i in range(min(5, len(t_obs))):
        prey_err = abs(float(pred[i, 0] - true[i, 0]))
        pred_err = abs(float(pred[i, 1] - true[i, 1]))
        print(
            f"{float(t_obs[i]):8.3f} | "
            f"{float(pred[i, 0]):12.6f} "
            f"{float(true[i, 0]):12.6f} "
            f"{prey_err:10.4e} | "
            f"{float(pred[i, 1]):12.6f} "
            f"{float(true[i, 1]):12.6f} "
            f"{pred_err:10.4e}"
        )


def print_vector_field_check(results):
    eps = 1e-8
    test_states = as_array(results.get("test_states", results["states"][:5]))
    nn_residual = as_array(
        results.get("sample_nn_residual", results["learned_residual"][: len(test_states)])
    )
    true_residual = as_array(
        results.get("sample_true_residual", results["true_residual"][: len(test_states)])
    )
    model_vf = as_array(
        results.get("sample_model_vf", results["model_vf"][: len(test_states)])
    )
    true_vf = as_array(
        results.get("sample_true_vf", results["true_vf"][: len(test_states)])
    )

    print("\nVector field check at sample states")
    print("-" * 125)
    print(
        f"{'state':>16} | "
        f"{'NN residual':>24} | "
        f"{'true residual':>24} | "
        f"{'res rel err':>12} | "
        f"{'model rhs':>24} | "
        f"{'true rhs':>24} | "
        f"{'rhs rel err':>12}"
    )
    print("-" * 125)

    for state, nn_res, true_res, model, true in zip(
        test_states,
        nn_residual,
        true_residual,
        model_vf,
        true_vf,
    ):
        res_rel = np.linalg.norm(nn_res - true_res) / (np.linalg.norm(true_res) + eps)
        rhs_rel = np.linalg.norm(model - true) / (np.linalg.norm(true) + eps)
        print(
            f"{str([float(state[0]), float(state[1])]):>16} | "
            f"{str([round(float(nn_res[0]), 4), round(float(nn_res[1]), 4)]):>24} | "
            f"{str([round(float(true_res[0]), 4), round(float(true_res[1]), 4)]):>24} | "
            f"{float(res_rel):12.4e} | "
            f"{str([round(float(model[0]), 4), round(float(model[1]), 4)]):>24} | "
            f"{str([round(float(true[0]), 4), round(float(true[1]), 4)]):>24} | "
            f"{float(rhs_rel):12.4e}"
        )


def print_summary(results):
    true_train = as_array(results["true_traj"])
    noisy_train = as_array(results.get("noisy_train_traj", true_train))
    pred_train = as_array(results["pred_traj"])
    true_val = as_array(results.get("true_validation_traj", true_train))
    pred_val = as_array(results.get("pred_validation_traj", pred_train))
    t_obs = get_time_grid(results, true_train)

    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"experiment type        : {results.get('experiment_type', 'unknown')}")
    print(f"solver type            : {results.get('solver_type', results.get('solver_method', results.get('method', 'rk4')))}")
    print(f"integration step size  : {float(results.get('integration_step_size', results.get('solver_step_size', results['h_model']))):.6f}")
    if "noise_level" in results or "train_noise_level" in results:
        print(f"noise level            : {float(results.get('noise_level', results.get('train_noise_level'))):.6f}")
    else:
        print("noise level            : N/A")
    print(f"ratio                  : {results.get('ratio', 'N/A')}")
    print(f"h_model                : {float(results['h_model']):.6f}")
    print(f"best loss              : {metric(results, 'best_loss')}")
    print(f"train clean loss       : {metric(results, 'train_clean_loss')}")
    print(f"train noisy loss       : {metric(results, 'train_noisy_loss')}")
    if "best_stage" in results:
        print(f"best stage             : {results['best_stage']}")
        print(f"best epoch             : {results.get('best_epoch', 'N/A')}")
        print(f"best NN lr             : {metric(results, 'best_nn_lr')}")
        print(f"best physics lr        : {metric(results, 'best_physics_lr')}")
        print(f"best residual scale    : {metric(results, 'best_residual_scale')}")
        print(f"best L2 weight         : {metric(results, 'best_l2_weight')}")
        print(f"best ortho weight      : {metric(results, 'best_ortho_weight')}")
        print(f"train L2 regularization: {metric(results, 'train_l2_regularization')}")
        print(
            "train ortho regularize : "
            f"{metric(results, 'train_orthogonality_regularization')}"
        )
        print(
            "regularized objective  : "
            f"{metric(results, 'train_regularized_objective')}"
        )
    print(f"validation loss        : {metric(results, 'validation_loss')}")
    if "extrapolation_start" in results:
        print(f"extrapolation starts at: t = {float(results['extrapolation_start']):.3f}")
    print(f"learned physics params : {results.get('learned_f_physics', 'N/A')}")
    print("=" * 80)

    print_trajectory_check("Train clean trajectory check", t_obs, pred_train, true_train)
    print_trajectory_check("Train noisy observation check", t_obs, pred_train, noisy_train)
    print_trajectory_check("Validation trajectory check", t_obs, pred_val, true_val)

    print("\nTrajectory metrics")
    print("-" * 80)
    print(f"train clean batch MSE       : {metric(results, 'train_clean_batch_mse')}")
    print(
        "train clean batch rel error : "
        f"{metric(results, 'train_clean_batch_rel_error')}"
    )
    print(f"train noisy batch MSE       : {metric(results, 'train_noisy_batch_mse')}")
    print(
        "train noisy batch rel error : "
        f"{metric(results, 'train_noisy_batch_rel_error')}"
    )
    print(f"validation batch MSE        : {metric(results, 'validation_batch_mse')}")
    print(
        "validation batch rel error  : "
        f"{metric(results, 'validation_batch_rel_error')}"
    )
    print(f"train clean sample MSE      : {metric(results, 'train_sample_clean_mse')}")
    print(
        "train clean sample rel error: "
        f"{metric(results, 'train_sample_clean_rel_error')}"
    )
    print(f"train noisy sample MSE      : {metric(results, 'train_sample_noisy_mse')}")
    print(
        "train noisy sample rel error: "
        f"{metric(results, 'train_sample_noisy_rel_error')}"
    )
    print(f"validation sample MSE       : {metric(results, 'validation_sample_mse')}")
    print(
        "validation sample rel error : "
        f"{metric(results, 'validation_sample_rel_error')}"
    )
    print(f"extrapolation sample MSE    : {metric(results, 'extrapolate_sample_mse')}")
    print(
        "extrapolation sample rel err: "
        f"{metric(results, 'extrapolate_sample_rel_error')}"
    )
    print(f"extrapolation batch MSE     : {metric(results, 'extrapolate_batch_mse')}")
    print(
        "extrapolation batch rel err : "
        f"{metric(results, 'extrapolate_batch_rel_error')}"
    )

    print("\nVector-field metrics on grid")
    print("-" * 80)
    print_vector_field_check(results)
    print("\nVector-field aggregate metrics")
    print("-" * 80)
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
        description="Visualize saved Lotka-Volterra residual Neural ODE results."
    )
    parser.add_argument(
        "result_path",
        nargs="?",
        default="wrong_physics_trainable.pkl",
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
