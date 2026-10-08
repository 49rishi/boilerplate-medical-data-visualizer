import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Import data
df = pd.read_csv("medical_examination.csv")


# Add overweight column
df["overweight"] = (
    df["weight"] / ((df["height"] / 100) ** 2) > 25
).astype(int)


# Normalize cholesterol
df["cholesterol"] = (df["cholesterol"] > 1).astype(int)


# Normalize glucose
df["gluc"] = (df["gluc"] > 1).astype(int)


# Draw categorical plot
def draw_cat_plot():

    # Create DataFrame for categorical plot
    df_cat = pd.melt(
        df,
        id_vars=["cardio"],
        value_vars=[
            "cholesterol",
            "gluc",
            "smoke",
            "alco",
            "active",
            "overweight"
        ]
    )

    # Group and count values
    df_cat = (
        df_cat
        .groupby(["cardio", "variable", "value"])
        .size()
        .reset_index(name="total")
    )

    # Create categorical plot
    cat_plot = sns.catplot(
        x="variable",
        y="total",
        hue="value",
        col="cardio",
        data=df_cat,
        kind="bar"
    )

    # Get figure
    fig = cat_plot.fig

    fig.savefig("catplot.png")

    return fig


# Draw heat map
def draw_heat_map():

    # Copy data
    df_heat = df.copy()

    # Clean the data
    df_heat = df_heat[
        (df_heat["ap_lo"] <= df_heat["ap_hi"])
        & (
            df_heat["height"]
            >= df_heat["height"].quantile(0.025)
        )
        & (
            df_heat["height"]
            <= df_heat["height"].quantile(0.975)
        )
        & (
            df_heat["weight"]
            >= df_heat["weight"].quantile(0.025)
        )
        & (
            df_heat["weight"]
            <= df_heat["weight"].quantile(0.975)
        )
    ]

    # Calculate correlation matrix
    corr = df_heat.corr()

    # Mask upper triangle
    mask = np.triu(
        np.ones_like(corr, dtype=bool)
    )

    # Set up matplotlib figure
    fig, ax = plt.subplots(
        figsize=(12, 12)
    )

    # Draw heatmap
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".1f",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        ax=ax
    )

    fig.savefig("heatmap.png")

    return fig