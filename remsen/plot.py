from typing import List

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pgf import FigureCanvasPgf
from matplotlib.backend_bases import register_backend

from remsen.training import tensorboard_dataframe


def configure_latex(scaler: float = 1.5):
    """Configure matplotlib for LaTeX figure output."""
    register_backend('pdf', FigureCanvasPgf)
    plt.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "pgf.preamble": [
                r"\usepackage[utf8x]{inputenc}",
                r"\usepackage[T1]{fontenc}",
            ],
            "figure.figsize": [scaler * 3.39, scaler * 2.0951352218621437],
            "text.usetex": True,
            "backend": "ps",
        }
    )

def plot_training(
    names: List[str],
    splits: List[str] = ["validation"],
    metric: str = "iou",
) -> None:
    """Plot training sequence from TensorBoard for given model/split/metric."""
    configure_latex()
    fig, ax = plt.subplots()
    ax.set_xlabel("$\mathrm{Epoch}$")

    if metric == "iou":
        ax.set_ylabel("$\mathrm{IoU}$")
    else:
        ax.set_ylabel(metric)

    for name in names:
        for split in splits:
            training_data = tensorboard_dataframe(name=name, split=split)
            metric_series = training_data["val_iou"]
            ax.plot(metric_series)

    fig.tight_layout()
    filename = (
        "+".join(sorted(names)) + "-" + "+".join(sorted(splits)) + "-" + metric
    )
    path = "/code/tex/img/metrics/" + filename + ".pdf"
    fig.savefig(path)
    return fig, ax
