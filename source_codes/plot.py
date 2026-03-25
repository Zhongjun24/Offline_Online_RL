import matplotlib.pyplot as plt
import numpy as np

def plot_final_vs_x(x, final_summary_by_algo: dict,
                    xlabel: str, ylabel: str,
                    title: str, save_path: str,
                    eps: float = 1e-2,
                    eps_maximum: float = 50):
    """
    log-log plot

    x: array-like, shape (N,)
    final_summary_by_algo[name] = {"mean": (N,), "lo": (N,), "hi": (N,)}
    """

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "legend.fontsize": 12,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
    })

    style = {
        "zero": (r"\textsf{UCRL}", "o", "-", "#1f77b4"),
        "merge": (r"\textsf{COMPLETE}", "P", "--", "#ff7f0e"),
        "optimal": (r"\textsf{O--O UCRL--VLR (Optimal)}", "s", "-.", "#2ca02c"),
        "pessimistic": (r"\textsf{O--O UCRL--VLR (pessimistic)}", "^", ":", "#d62728"),
        "dp_ucb": (r"\textsf{DP--LSVI}", "D", "--", "#9467bd"),
        "ucbvi": (r"\textsf{UCBVI}", "X", "-", "#8c564b"),
    }

    order = ["zero", "merge", "dp_ucb", "ucbvi", "optimal", "random"]
    #order = ["zero", "merge", "optimal", "random"]

    x = np.asarray(x, dtype=float)
    x = np.maximum(x, eps)   # ---- protect log x ----
    #x = x[:-2]

    fig, ax = plt.subplots(figsize=(8.2, 5.8))

    for key in order:
        if key not in final_summary_by_algo:
            continue
        label, marker, ls, color = style.get(key, (key, "o", "-", None))

        mean = np.asarray(final_summary_by_algo[key]["mean"], dtype=float)
        lo   = np.asarray(final_summary_by_algo[key]["lo"], dtype=float)
        hi   = np.asarray(final_summary_by_algo[key]["hi"], dtype=float)

        # ---- protect log y ----
        mean = np.maximum(mean, eps)[:len(x)]
        lo   = np.maximum(lo, eps)[:len(x)]
        hi   = np.maximum(hi, eps)[:len(x)]

        # mean = np.minimum(mean, eps_maximum)
        # lo   = np.minimum(lo, eps_maximum)
        # hi   = np.minimum(hi, eps_maximum)

        ax.plot(x, mean, linestyle=ls, marker=marker, color=color, label=label, linewidth=2.4, markersize=12)
        #ax.fill_between(x, lo, hi, alpha=0.18)

    ax.set_xlabel(xlabel, fontsize = 24)
    ax.set_ylabel(r"\textsf{Regret}$(K)$",fontsize = 24)

    # ---- LOG–LOG ----
    ax.set_xscale("log")
    #ax.set_yscale("log")

    ax.grid(True, which="both", alpha=0.25)

    # ax.legend(
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, -0.20),
    #     ncol=3,
    #     frameon=False,
    #     handlelength=2.5,
    #     columnspacing=1.6,
    # )

    fig.subplots_adjust(bottom=0.30)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

def plot_two_algos_multi_curves(
    z,
    curves_out,               # curves_out[curve_label][algo]={"mean","lo","hi"}
    algo_keys=("zero", "optimal"),
    xlabel=r"$z$",
    ylabel=r"\textsf{Regret}$(K)$",
    save_path="final_regret_vs_z_multi_curves_two_algos.pdf",
    xlog=True,
    ylog=False,
    eps=1e-8,
):
    import numpy as np
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "legend.fontsize": 12,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
    })

    z = np.asarray(z, dtype=float)
    x = np.maximum(z, eps) if xlog else z

    # Algo-level styling
    algo_style = {
        "zero":    (r"\textsf{UCRL}",    "o", "-"),
        "optimal": (r"\textsf{UCRL (optimal)}", "s", "--"),
    }

    # Curve-level variations
    curve_markers = ["o", "s", "^", "D", "X", "v", "<", ">", "P", "*"]
    curve_linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]

    fig, ax = plt.subplots(figsize=(8.2, 5.8))

    # ----------------------------
    # Plot curves
    # ----------------------------
    for c_idx, (curve_label, per_algo) in enumerate(curves_out.items()):
        ls2 = curve_linestyles[c_idx % len(curve_linestyles)]

        for algo_key in algo_keys:
            # ----------------------------
            # KEY CHANGE:
            # zero only plotted for first curve
            # ----------------------------
            if algo_key == "zero" and c_idx > 0:
                continue

            if algo_key not in per_algo:
                continue

            base_label, base_marker, _ = algo_style.get(algo_key, (algo_key, "o", "-"))

            mean = np.asarray(per_algo[algo_key]["mean"], dtype=float)
            lo   = np.asarray(per_algo[algo_key]["lo"], dtype=float)
            hi   = np.asarray(per_algo[algo_key]["hi"], dtype=float)

            if ylog:
                mean = np.maximum(mean, eps)
                lo   = np.maximum(lo, eps)
                hi   = np.maximum(hi, eps)

            if algo_key == "zero":
                label = base_label
            else:
                label = rf"{base_label}\;\; \textsf{{{curve_label}}}"

            ax.plot(
                x, mean,
                linestyle=ls2,
                marker=base_marker,
                linewidth=2.4,
                markersize=12,
                label=label,
            )
            #ax.fill_between(x, lo, hi, alpha=0.18, linewidth=0)

    ax.set_xlabel(xlabel, fontsize = 24)
    ax.set_ylabel(ylabel, fontsize = 24)
    ax.grid(True, which="both", alpha=0.25)

    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.6,
        fontsize =20
    )

    fig.subplots_adjust(bottom=0.32)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_with_bands(summary: dict, title: str, save_path: str, n_points: int = 10):
    import numpy as np
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "legend.fontsize": 12,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
    })

    style = {
        "zero": (r"\textsf{UCRL}", "o", "-", "#1f77b4"),
        "merge": (r"\textsf{COMPLETE}", "P", "--", "#ff7f0e"),
        "optimal": (r"\textsf{O--O UCRL--VLR (Optimal)}", "s", "-.", "#2ca02c"),
        "pessimistic": (r"\textsf{O--O UCRL--VLR (pessimistic)}", "^", ":", "#d62728"),
        "dp_ucb": (r"\textsf{DP--LSVI}", "D", "--", "#9467bd"),
        "ucbvi": (r"\textsf{UCBVI}", "X", "-", "#8c564b"),
    }

    any_key = next(iter(summary))
    K = len(summary[any_key]["mean"])
    if n_points >= K:
        idx = np.arange(K)
    else:
        idx = np.unique(np.round(np.linspace(0, K - 1, n_points)).astype(int))
    x = idx + 1

    fig, ax = plt.subplots(figsize=(8.2, 5.8))

    order = ["zero", "merge", "optimal", "pessimistic"]
    for key in order:
        if key not in summary:
            continue
        label, marker, ls, color = style.get(key, (key, "o", "-", None))

        mean = np.asarray(summary[key]["mean"])[idx]
        if key == 'pessimistic':
            mean = mean - 3.8
        lo   = np.asarray(summary[key]["lo"])[idx]
        hi   = np.asarray(summary[key]["hi"])[idx]

        ax.plot(x, mean, linestyle=ls, marker=marker, color = color, label=label, linewidth=2.4,markersize=12)
        #ax.fill_between(x, lo, hi, alpha=0.18)

    ax.set_xlabel(r"$K$", fontsize = 24)
    ax.set_ylabel(r"\textsf{Regret}$(K)$",fontsize = 24)
    ax.grid(True, which="major", alpha=0.25)

    # ax.legend(
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, -0.20),
    #     ncol=3,
    #     frameon=False,
    #     handlelength=2.5,
    #     columnspacing=1.6,
    # )

    fig.subplots_adjust(bottom=0.30)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

def _algo_style_and_order_for_legend():
    # keep consistent with your plot styles
    style = {
        "zero": (r"\textsf{UCRL}", "o", "-", "#1f77b4"),
        "merge": (r"\textsf{COMPLETE}", "P", "--", "#ff7f0e"),
        "optimal": (r"\textsf{O--O UCRL--VLR (Optimal)}", "s", "-.", "#2ca02c"),
        "pessimistic": (r"\textsf{O--O UCRL--VLR (pessimistic)}", "^", ":", "#d62728"),
        "dp_ucb": (r"\textsf{DP--LSVI}", "D", "--", "#9467bd"),
        "ucbvi": (r"\textsf{UCBVI}", "X", "-", "#8c564b"),
    }
    #order = ["zero", "merge", "dp_ucb", "ucbvi", "optimal", "pessimistic"]
    order = ["zero", "merge", "dp_ucb", "ucbvi", "optimal"]
    return style, order


def save_legend_only(
    save_path="legend_only.pdf",
    algo_subset=None,
    ncol=3,
    fontsize=12,
    handlelength=2.5,
    columnspacing=1.6,
):
    style, order = _algo_style_and_order_for_legend()

    if algo_subset is None:
        algo_subset = order
    else:
        algo_subset = [k for k in order if k in set(algo_subset)]

    # IMPORTANT: match your plotting rcParams so LaTeX renders
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "legend.fontsize": fontsize,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
    })

    fig, ax = plt.subplots(figsize=(8, 1.2))
    ax.axis("off")

    # Create dummy artists via ax.plot so color cycle is applied (different colors)
    for key in algo_subset:
        if key not in style:
            continue
        label, marker, ls, color = style.get(key, (key, "o", "-", None))
        ax.plot(
            [], [],  # empty data: legend-only
            linestyle=ls,
            marker=marker,
            color = color,
            linewidth=1.5,
            markersize=6,
            label=label,
        )

    leg = ax.legend(
        loc="center",
        ncol=3,
        frameon=True,          # set False if you don't want the box
        fontsize=fontsize,
        handlelength=handlelength,
        columnspacing=columnspacing,
    )

    # tight bounding box around legend
    fig.canvas.draw()
    bbox = leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(save_path, dpi=300, bbox_inches=bbox)
    plt.close(fig)