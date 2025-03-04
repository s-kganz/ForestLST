'''
This file holds plotting functions from eartmover.ai's demo of 
a performant torch training loop. Copied from

'''

import matplotlib.pyplot as plt
import json
import numpy as np
import seaborn

def parse_log(fname):
    with open(fname) as f:
        lines = f.readlines()

    messages = []
    for line in lines:
        try:
            messages.append(json.loads(line.strip()))
        except json.JSONDecodeError:
            pass
    return messages


def plot_wait_time(messages, ax, title="Time waiting"):
    if title:
        ax.set_title(title)

    wait_times = []
    end = None
    in_training = True
    for m in messages:
        if m["event"] == "epoch end":
            in_training = False
        if m["event"] == "epoch":
            in_training = True
        if m["event"] == "training end":
            end = m["time"]
        if m["event"] == "training start" and end is not None and in_training:
            wait_times.append(m["time"] - end)

    wait_times = np.array(wait_times)
    max_show = wait_times.mean() * 3

    print("average wait time", wait_times.mean())

    ax.hist(wait_times, bins=np.linspace(0, max_show, 100), color="#6D0EDB")[-1]
    ax.set_xlabel("time (sec)")


def plot_log(messages, ax, title=""):
    origin = messages[0]["time"]
    assert messages[0]["event"] == "run start"

    rows = {"setup": 4, "get-training-batch": 3, "get-validation-batch": 2, "train": 1, "epoch": 0}

    ax.set_yticks(list(rows.values()), labels=list(rows))

    if title:
        ax.set_title(title)

    data = {"batches": []}

    for m in messages:
        t = m["time"] - origin

        if m["event"] == "setup end":
            ax.barh(
                rows["setup"],
                m["duration"],
                left=t - m["duration"],
                edgecolor="k",
                linewidth=0.1,
                color="#6D0EDB",
                zorder=1,
            )

        if m["event"] == "get-training-batch end":
            ax.barh(
                rows["get-training-batch"],
                m["duration"],
                left=t - m["duration"],
                # alpha=0.5,
                color="#C396F9",
                zorder=1,
            )
            data["batches"].append(m["duration"])

        if m["event"] == "get-validation-batch end":
            ax.barh(
                rows["get-validation-batch"],
                m["duration"],
                left=t - m["duration"],
                # alpha=0.5,
                color="#C396F9",
                zorder=1,
            )
            #data["batches"].append(m["duration"])

        if m["event"] == "training end":
            ax.barh(
                rows["train"],
                m["duration"],
                left=t - m["duration"],
                color="#FF6554",
                zorder=1,  # edgecolor="k", linewidth=0.1,
            )

        if m["event"] == "epoch end":
            ax.barh(
                rows["epoch"],
                m["duration"],
                left=t - m["duration"],
                edgecolor="k",
                linewidth=0.1,
                color="#FF9E0D",
                zorder=1,
            )

    ax.grid(axis="x", zorder=0, alpha=0.5)
    ax.set_xlabel("time (sec)")

    print("average batch duration", np.mean(data["batches"]))


def plot(fname):
    messages = parse_log(fname)
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(16, 6), width_ratios=[3, 1], dpi=400)

    plot_log(messages, axes[0], title="")
    plot_wait_time(messages, axes[1], title="")

    for m in messages:
        if m["event"] == "run start":
            # text_str = "\n".join([f"{k}: {v}" for k, v in m["locals"].items() if v is not None])
            props = dict(boxstyle="round", facecolor="#F5F5F5", alpha=0.5)
            fig.text(
                0.5,
                -0.03,
                "",
                fontsize=14,
                horizontalalignment="center",
                verticalalignment="top",
                bbox=props,
            )
            break

def plot_learning_curve(history_path: str, ax: plt.Axes | None, **kwargs) -> list | None:
    try:
        from .training import parse_tensorboard
    except ImportError:
        raise RuntimeError("Module util.training must be loaded to use plot_learning_curve()")

    losses = parse_tensorboard(history_path, ["Loss/train", "Loss/valid"])
    epochs = np.arange(1, len(losses["Loss/train"]["value"])+1)

    if ax is not None:
        return [
            plt.plot(epochs, losses["Loss/train"]["value"], label="Train", **kwargs),
            plt.plot(epochs, losses["Loss/valid"]["value"], label="Valid", **kwargs)
        ]
    else:
        ax.plot(epochs, losses["Loss/train"]["value"], label="Train", **kwargs)
        ax.plot(epochs, losses["Loss/valid"]["value"], label="Valid", **kwargs)

def plot_all_scalars_in_run(history, subplots_kw=dict()):
    try:
        from util.training import parse_tensorboard
    except ImportError:
        raise RuntimeError("Module util.training must be available to use plot_all_scalars_in_run")
    
    all_scalars = parse_tensorboard(history)
    keys = set(k.split("/")[0] for k in all_scalars.keys())
    fig, ax = plt.subplots(int(np.ceil(len(keys) / 3)), 3, sharex=True, sharey=False, **subplots_kw)
    
    for (k, a) in zip(keys, ax.flatten()):
        a.set_title(k)
        for kk in all_scalars:
            if kk.startswith(k):
                label = kk.split("/")[-1] if "/" in kk else ""
                a.plot(all_scalars[kk]["step"], all_scalars[kk]["value"], label=label)
    
    lines, labels = [], []
    for ax in fig.axes:
        if ax.has_data():
            ax_lines, ax_labels = ax.get_legend_handles_labels()
            for li, la in zip(ax_lines, ax_labels):
                if la not in labels:
                    lines.append(li)
                    labels.append(la)
        else:
            ax.axis("off")
    
    fig.supxlabel("Epoch")
    fig.legend(lines, labels)
    fig.tight_layout()
    
    return fig, ax
    