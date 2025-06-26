import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from copy import deepcopy



def plot_finetune_metrics(
    df_finetune: pd.DataFrame, 
    train_loss: bool, 
    val_metric: bool, 
    monitor_val_metric: bool,
    name_val_metric: str,
    name_monitor_val_metric: str | None,
    best_step: int,
    title: str = "Finetune Metrics"
) -> Axes:
    '''
    Utility to plot the summary finetune metrics.

    Parameters:
        df_finetune (pd.DataFrame): 
            The dataframe obtained from "collect_finetune_attrs" method.
            No check is done on this.
        train_loss (bool):
            Whether to plot the train loss info.
        val_metric (bool):
            Whether to plot the validation metric info.
        monitor_val_metric (bool):
            Whether to plot the monitor validation metric info.
        name_val_metric (str):
            Name of the validation metric displayed in the plot.
        name_monitor_val_metric (str):
            Name of the monitor validation metric displayed in the plot.
        best_step (int):
            Best finetune step info.
        title (str, optional):
            Plot title.

    Returns:
        Axes: The plot.
    '''
    if not train_loss and not val_metric and not monitor_val_metric:
        raise ValueError((
            "At least one finetune metric between"
            "train_loss, val_metric and monitor_val_metric must be set to True."
        ))
    
    df_copy = deepcopy(df_finetune)
    is_monitor_info_available = not np.isnan(df_copy["val_monitor_metric"][0])
    
    if not is_monitor_info_available and monitor_val_metric:
        raise ValueError("The monitor validation metric is not available.")
    
    columns_to_melt = []
    if train_loss: columns_to_melt.append("train_loss")
    if val_metric: columns_to_melt.append("val_metric")
    if monitor_val_metric: columns_to_melt.append("val_monitor_metric")

    df_long = pd.melt(
        df_copy, 
        id_vars="step", 
        value_vars=columns_to_melt,
        var_name="metric",
        value_name="value"
    )

    label_val_metric = f"{name_val_metric} (early stop)"
    label_monitor_val_metric = f"{name_monitor_val_metric} (monitored)"

    name_map = {
        "train_loss": "Train Loss",
        "val_metric": f"{label_val_metric}",
        "val_monitor_metric": f"{label_monitor_val_metric}"
    }

    df_long["metric"] = df_long["metric"].map(name_map)

    fig, ax = plt.subplots(figsize=(8, 8))

    palette = {
        "Train Loss": "blue", 
        f"{label_val_metric}": "orange", 
        f"{label_monitor_val_metric}": "green"
    }
    
    sns.lineplot(
        data=df_long,
        x="step",
        y="value",
        hue="metric",
        palette=palette,
        ax=ax,
        linewidth=3
    )

    ax.axvline(
        x=best_step,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Best Step",
    )
    
    ax.legend(title="Legend")
    ax.set_title(title)
    return ax
