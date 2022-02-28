import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TAGS = [
    "Train/Loss",
    "Train/Metric",
    "Test/Loss",
    "Test/Metric"
]

FILES_NAMES = {
    "Train/Loss": "train-loss.pdf",
    "Train/Metric": "train-acc.pdf",
    "Test/Loss": "test-loss.pdf",
    "Test/Metric": "test-acc.pdf"}

AXE_LABELS = {
    "Train/Loss": "Train loss",
    "Train/Metric": "Train acc",
    "Test/Loss": "Test loss",
    "Test/Metric": "Test acc"
}

LEGEND = {
    "local": "Local",
    "clustered": "Clustered FL",
    "FedAvg": "FedAvg",
}

MARKERS = {
    "local": "x",
    "clustered": "s",
    "FedAvg": "h",
}

COLORS = {
    "local": "tab:blue",
    "clustered": "tab:orange",
    "FedAvg": "tab:green"
}


def make_plot(logs_dir_, tag_, save_path):
    """
    :param logs_dir_: path of the logs directory,  should contain sub-directories corresponding to methods
        each sub-directory must contain a single tf events file.
    :param tag_: the tag to be plotted, possible are "Train/Loss", "Train/Metric", "Test/Loss", "Test/Metric"
    :param save_path: path to save the resulting plot
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    for method in os.listdir(logs_dir_):

        method_path = os.path.join(logs_dir_, method)

        for task in os.listdir(method_path):
            if task == "global":
                task_path = os.path.join(method_path, task)
                ea = EventAccumulator(task_path).Reload()

                tag_values = []
                steps = []
                for event in ea.Scalars(tag_):
                    tag_values.append(event.value)
                    steps.append(event.step)

                if method in LEGEND:
                    ax.plot(
                        steps,
                        tag_values,
                        linewidth=5.0,
                        marker=MARKERS[method],
                        markersize=20,
                        markeredgewidth=5,
                        label=f"{LEGEND[method]}",
                        color=COLORS[method]
                    )

    ax.grid(True, linewidth=2)

    ax.set_ylabel(AXE_LABELS[tag_], fontsize=50)
    ax.set_xlabel("Rounds", fontsize=50)

    ax.tick_params(axis='both', labelsize=25)
    ax.legend(fontsize=30)

    os.makedirs(save_path, exist_ok=True)
    fig_path = os.path.join(save_path, f"{FILES_NAMES[tag_]}")
    plt.savefig(fig_path, bbox_inches='tight')
