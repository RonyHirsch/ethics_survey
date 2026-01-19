"""
Visualization functions for manuscript figures and supplementary materials.

Generates publication-quality plots (histograms, scatter plots, stacked bars, pie charts) from processed survey
data for the main manuscript and supplementary materials.

Author: RonyHirsch
"""

import os
import re
import math
import warnings
from functools import reduce
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns
from PIL import Image
from matplotlib.colors import to_rgb
import textwrap
import analyze_survey
import survey_mapping
import plotter

max_text_width = 20  # characters per line
LEGEND_FONT = 17

AGE_BINS = [18, 25, 35, 45, 55, 65, 75, 120]
AGE_LABELS = ["18-25", "26-35", "36-45", "46-55", "56-65", "66-75", "76+"]

SAMPLE_COLORS = {"exploratory": "#4097AA",
                 "pre-registered": "#2C7695",
                 "follow-up": "#D1A87E"}

SAMPLE_COLORS_RATINGS = {"exploratory": {1: "#B5D7DF",
                                         2: "#7DBFCC",
                                         3: "#4097AA",
                                         4: "#2A7485",
                                         5: "#1A5563"},
                         "pre-registered": {1: "#A3B8C3",
                                            2: "#6C8A9B",
                                            3: "#2C7695",
                                            4: "#214D72",
                                            5: "#1A3A55"},
                         "follow-up": {1: "#F3DECA",
                                       2: "#E4C6A7",
                                       3: "#D1A87E",
                                       4: "#B8875A",
                                       5: "#966840"}}

SAMPLE_COLORS_SELECTED_NO = {"exploratory": {0: SAMPLE_COLORS_RATINGS["exploratory"][1],
                                             1: SAMPLE_COLORS_RATINGS["exploratory"][3]},
                             "pre-registered": {0: SAMPLE_COLORS_RATINGS["pre-registered"][1],
                                                1: SAMPLE_COLORS_RATINGS["pre-registered"][3]}}

GENDER_COLORS = {"Female": "#d4a373",
                 "Male": "#4a5759",
                 "Non-binary": "#f7e1d7",
                 "Genderqueer": "#edafb8",
                 "Genderfluid": "#9d8189",
                 "Prefer not to say": "#dedbd2"}

EDU_COLORS = {survey_mapping.EDU_NONE: "#DCEDFF",
              survey_mapping.EDU_PRIM: "#90C3C8",
              survey_mapping.EDU_SECD: "#759FBC",
              survey_mapping.EDU_POSTSEC: "#1F5673",
              survey_mapping.EDU_GRAD: "#463730"}


YESNO_COLORS = {survey_mapping.ANS_YES: "#3C5968", survey_mapping.ANS_NO: "#B53B03"}

C_I_HOW_COLOR_MAP = {survey_mapping.ANS_C_NECESSARY: "#1a4e73",
                     survey_mapping.ANS_I_NECESSARY: "#346182",
                     survey_mapping.ANS_SAME: "#013a63",
                     survey_mapping.ANS_THIRD: "#4d7592",
                     survey_mapping.ANS_NO: "#B53B03"}
C_I_HOW_ORDER = [survey_mapping.ANS_NO, survey_mapping.ANS_SAME, survey_mapping.ANS_C_NECESSARY,
                 survey_mapping.ANS_I_NECESSARY, survey_mapping.ANS_THIRD]

MAJORITY_COLORS = {"majority": "#155665", "minority": "#C4CFCE"}
# orig {"majority": "#33546D", "minority": "#F9F4E6"}

EARTH_DANGER_CLUSTER_COLORS_MAJORITY = {0: "#FFBE80", 1: "#9B532D"}


def unify_data_all(exploratory_path, preregistered_path, followup_path, save_path):
    # I already have all the demographics pre-processed, and age group
    sub_df_exp = pd.read_csv(os.path.join(exploratory_path, "sub_df.csv"))
    sub_df_exp["sample"] = "exploratory"

    sub_df_pre = pd.read_csv(os.path.join(preregistered_path, "sub_df.csv"))
    sub_df_pre["sample"] = "pre-registered"

    sub_df_fu = pd.read_csv(os.path.join(followup_path, "sub_df.csv"))
    sub_df_fu["sample"] = "follow-up"

    samples = {"exploratory": sub_df_exp,
               "pre-registered": sub_df_pre,
               "follow-up": sub_df_fu}
    main_survey = {"exploratory": exploratory_path, "pre-registered": preregistered_path}

    # stuff I need to add to the follow-up:
    sub_df_fu["source"] = "Prolific"  # source: all of them were paid
    # age group
    sub_df_fu["age_group"] = pd.cut(sub_df_fu["How old are you?"], bins=AGE_BINS, labels=AGE_LABELS, include_lowest=True).astype(str)

    # stuff I need to add: ICS group, and EiD clusters
    ics = "i_c_s"
    eid = "earth_danger"
    sub_id = "response_id"

    combine_list = list()

    for sample in main_survey:
        df = samples[sample]
        ics_df = pd.read_csv(os.path.join(main_survey[sample], ics, "ics_with_c_groups.csv"))
        eid_df = pd.read_csv(os.path.join(main_survey[sample], eid, "earth_danger_clusters.csv"))
        # need to be careful when mapping clusters to meanings here
        if sample == "exploratory":  # Cluster 0 is the anthropocentric one and Cluster 1 is the non
            eid_df["Cluster"] = eid_df["Cluster"].map({0: "anthropocentric", 1: "non-anthropocentric"})
        else:  # in the prereg it's the opposite: Cluster 0 is the non, and Cluster 1 is the anthropocentric
            eid_df["Cluster"] = eid_df["Cluster"].map({0: "non-anthropocentric", 1: "anthropocentric"})
        dfs = [df, eid_df, ics_df]
        merged = reduce(lambda left, right: left.merge(right, on=sub_id, how="left"), dfs)
        combine_list.append(merged)

    combined = pd.concat(combine_list + [sub_df_fu], axis=0, ignore_index=True)
    combined.to_csv(os.path.join(save_path, "sub_df.csv"), index=False)

    return combined


def plot_hist(df, save_path, save_name, fmt, x_col, y_col, y_order, title, y_title, x_title,
              size_inches_x, size_inches_y, alpha_val, color_palette, stat_type, element, multiple_resolve,
              bins, x_ticks, y_ticks, x_tick_rotation=0, bar_shrink=1.0):

    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    plt.figure(figsize=(8, 5))
    sns.despine(top=True, right=True)

    # check if x is numeric or categorical
    from pandas.api.types import is_numeric_dtype
    x_is_numeric = is_numeric_dtype(df[x_col])
    df_plot = df.copy()
    if not x_is_numeric:
        df_plot[x_col] = pd.Categorical(df_plot[x_col].astype("string"), categories=x_ticks, ordered=True)

    ax = sns.histplot(
        data=df_plot,
        x=x_col,
        hue=y_col,
        hue_order=y_order,
        stat=stat_type,  # y-axis in % (percent), density, count
        common_norm=False,  # 100% per group - not common
        multiple=multiple_resolve,  # "layer": overlay groups, "dodge"=side by side
        element=element,  # "poly": filled polygon (no bars), "step": bars overlapping, “bars”: bars
        fill=True,  # fill below the line
        bins=bins if x_is_numeric else None,
        alpha=alpha_val,
        linewidth=2,
        palette=color_palette,
        discrete=False if x_is_numeric else True,  # hint to seaborn for categorical x
        shrink=1.0 if x_is_numeric else bar_shrink
    )

    # titles
    plt.title(title, fontsize=25)
    plt.xlabel(x_title, fontsize=22, labelpad=10)
    plt.ylabel(y_title, fontsize=22, labelpad=8)
    plt.tight_layout()

    # ticks
    plt.yticks(y_ticks, fontsize=18)
    plt.xticks(x_ticks, fontsize=18)

    # x tick
    from textwrap import fill
    ax = plt.gca()
    current_labels = [t.get_text() for t in ax.get_xticklabels()]
    wrapped = [
        fill(lbl, width=max_text_width, break_long_words=True) if len(lbl) > max_text_width else lbl
        for lbl in current_labels
    ]
    ax.set_xticklabels(wrapped, fontsize=18)

    # tilt x
    if x_tick_rotation > 0:
        ax = plt.gca()
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(x_tick_rotation)
            lbl.set_horizontalalignment("right")  # end of string at tick
            lbl.set_verticalalignment("top")  # keeps it tidy below axis
            lbl.set_rotation_mode("anchor")  # anchor stays on the tick
        ax.tick_params(axis="x", pad=6)  # a bit more breathing room

    # legend
    present = pd.Index(df[y_col].unique())
    legend_levels = [lvl for lvl in SAMPLE_COLORS.keys() if (lvl in present) and (lvl in SAMPLE_COLORS)]
    proxy_handles = [
        Line2D([0], [0],
               marker='o', linestyle='None',
               markerfacecolor=SAMPLE_COLORS[lvl],
               markeredgecolor='none', markersize=15)
        for lvl in legend_levels
    ]
    proxy_labels = [str(lvl).title() for lvl in legend_levels]
    ax.legend(proxy_handles, proxy_labels,
              title="", title_fontsize=18, fontsize=LEGEND_FONT,
              frameon=False, handlelength=1.0, handletextpad=0.5, borderpad=0.2)

    # save plot
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(size_inches_x, size_inches_y)
    plt.savefig(os.path.join(save_path, f"{save_name}.{fmt}"), format=fmt, dpi=1000, bbox_inches="tight", pad_inches=0.01)
    del figure
    plt.clf()
    plt.close()

    # csv to describe it
    if x_is_numeric:
        g = df.groupby(y_col, dropna=False)[x_col]
        result = g.agg(n_valid="count", mean="mean", sd="std", min="min", max="max",
                       q1=lambda s: s.quantile(0.25), median="median", q3=lambda s: s.quantile(0.75)).reset_index()
    else:
        df_plot[x_col] = df_plot[x_col].astype("string").fillna("NaN")
        ct = (df_plot.groupby([y_col, x_col], dropna=False).size().reset_index(name="count"))
        result = ct.pivot(index=y_col, columns=x_col, values="count").fillna(0).reset_index()
    result.to_csv(os.path.join(save_path, f"{save_name}.csv"), index=False)
    return


def plot_hist_multiselect(df, save_path, save_name, fmt, x_col, y_col, y_order, title,
                          y_title, x_title, n_col, denom_col, color_palette, metric="percent",
                          value_col=None, x_ticks=None, y_ticks=None, x_tick_rotation=0, bar_shrink=0.8,
                          size_inches_x=13, size_inches_y=8, alpha_val=1.0, x_text_width=max_text_width,
                          title_fontsize=25, y_tick_fontsize=24, x_tick_fontsize=20,  # NEW
                          ylabel_fontsize=22, xlabel_fontsize=22, legend_fontsize=16):  # NEW
    df_plot = df.copy()
    if value_col is None:
        if metric == "percent":
            if (n_col is None) or (denom_col is None):
                raise ValueError("For metric='percent', provide either value_col or both n_col and denom_col.")
            df_plot = df_plot.assign(_value=(df_plot[n_col] / df_plot[denom_col] * 100.0))
            value_col = "_value"
        elif metric == "count":
            if n_col is None:
                raise ValueError("For metric='count', provide n_col or value_col.")
            value_col = n_col

    if x_ticks is None:
        # default: order topics by total value across y groups (descending)
        x_order = (df_plot.groupby(x_col)[value_col].sum()
                   .sort_values(ascending=False).index.tolist())
    else:
        x_order = list(x_ticks)

    df_plot[x_col] = pd.Categorical(df_plot[x_col].astype("string"), categories=x_order, ordered=True)

    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    plt.figure(figsize=(8, 5))
    sns.despine(top=True, right=True)

    ax = sns.barplot(data=df_plot, x=x_col, y=value_col, hue=y_col, hue_order=y_order, dodge=True, errorbar=None,
                     estimator=sum, palette=color_palette, width=bar_shrink)
    for p in ax.patches:
        p.set_alpha(alpha_val)

    # axis
    plt.title(title, fontsize=title_fontsize)  # CHANGED
    plt.xlabel(x_title, fontsize=xlabel_fontsize, labelpad=10)  # CHANGED
    plt.ylabel(y_title, fontsize=ylabel_fontsize, labelpad=8)  # CHANGED
    plt.tight_layout()

    if y_ticks is not None:
        plt.yticks(y_ticks, fontsize=y_tick_fontsize)  # CHANGED
    else:
        plt.yticks(fontsize=y_tick_fontsize)  # CHANGED

    if x_ticks is not None:
        plt.xticks(x_order, fontsize=x_tick_fontsize)  # CHANGED
    else:
        plt.xticks(fontsize=x_tick_fontsize)  # CHANGED

    # text labels
    from textwrap import fill
    ax = plt.gca()
    current_labels = [t.get_text() for t in ax.get_xticklabels()]
    cleaned_labels = [re.sub(r"\s*\([^)]*\)", "", str(lbl)).strip() for lbl in current_labels]
    wrapped = [fill(lbl, width=x_text_width, break_long_words=True) if len(lbl) > x_text_width else lbl for lbl in cleaned_labels]
    ax.set_xticklabels(wrapped, fontsize=x_tick_fontsize)  # CHANGED

    if x_tick_rotation > 0:
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(x_tick_rotation)
            lbl.set_horizontalalignment("right")
            lbl.set_verticalalignment("top")
            lbl.set_rotation_mode("anchor")
        ax.tick_params(axis="x", pad=6)

    # legend
    present = pd.Index(df_plot[y_col].unique())
    legend_levels = [lvl for lvl in color_palette.keys() if (lvl in present) and (lvl in color_palette)]
    proxy_handles = [
        Line2D([0], [0], marker='o', linestyle='None',
               markerfacecolor=color_palette[lvl],
               markeredgecolor='none', markersize=15)
        for lvl in legend_levels
    ]
    proxy_labels = [str(lvl).title() for lvl in legend_levels]
    ax.legend(proxy_handles, proxy_labels,
              title="", title_fontsize=title_fontsize, fontsize=legend_fontsize,
              frameon=False, handlelength=1.0, handletextpad=0.5, borderpad=0.2)

    # save
    figure = plt.gcf()
    figure.set_size_inches(size_inches_x, size_inches_y)
    plt.savefig(os.path.join(save_path, f"{save_name}.{fmt}"), format=fmt, dpi=1000, bbox_inches="tight", pad_inches=0.01)
    del figure
    plt.clf()
    plt.close()

    # save csv with data
    df_out = df_plot.copy()
    df_out[x_col] = df_out[x_col].astype("string").fillna("NaN")
    result = (df_out
              .pivot_table(index=y_col, columns=x_col, values=value_col, aggfunc="sum")
              .fillna(0)
              .reset_index())
    result.to_csv(os.path.join(save_path, f"{save_name}.csv"), index=False)
    return


def plot_pie(df, save_path, save_name, fmt, size_inches_x, size_inches_y, color_palette,
             split_col, split_order, stat_col, stat_order, min_pct_to_label=1.5, text_width=max_text_width,
             annot_fontsize=20, title_fontsize=22, legend_fontsize=17, title_pad=1,
             shared_title=None, show_subplot_titles=True, show_legend=True,
             show_slice_labels=False):

    pies = []
    for grp in split_order:
        sub = df.loc[df[split_col] == grp, stat_col].dropna()
        if len(sub) == 0:
            # No data in this group -> zero slices
            props = pd.Series(0, index=stat_order, dtype=float)
        else:
            props = (
                sub.astype("category")
                .cat.set_categories(stat_order)
                .value_counts(normalize=True)
                .sort_index()
            )
        pies.append(props)

    n = len(split_order)
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n + 2.0, 6), constrained_layout=True)

    if n == 1:  # keep code robust if split_order has 1 item
        axes = [axes]

    wedges_all = []
    for ax, title, props in zip(axes, split_order, pies):
        values = props.values
        # colors aligned with gender_levels
        colors = [color_palette[g] for g in stat_order]
        # autopct only for non-zero totals
        if values.sum() <= 0:
            # Draw an empty circle with note
            ax.pie([1], colors=["#f0f0f0"], radius=1.0)
            ax.text(0, 0, "No data", ha="center", va="center", fontsize=annot_fontsize)
            ax.set_title(title, fontsize=annot_fontsize, pad=5, color="black")
            ax.set_aspect("equal")
            wedges_all.append([])
            continue

        def autopct_func(pct):
            if pct < min_pct_to_label:
                return ""
            return f"{pct:.0f}%" if pct >= 1 else f"{pct:.1f}%"

        wedges, texts = ax.pie(
            values,
            labels=None,  # We'll add labels manually
            colors=colors,
            startangle=90,
            counterclock=False,
            autopct=None,  # We'll add percentages manually
            wedgeprops={"linewidth": 1, "edgecolor": "white"},
            textprops={"size": annot_fontsize}
        )

        # add slice labels and percentages manually
        total = float(np.nansum(values))
        for i, (wedge, val, label) in enumerate(zip(wedges, values, stat_order)):
            ang = (wedge.theta2 + wedge.theta1) / 2
            ang_rad = np.deg2rad(ang)

            x = 0.6 * np.cos(ang_rad)
            y = 0.6 * np.sin(ang_rad)

            text_color = get_text_color(color_palette[label])
            pct = val / total * 100 if total > 0 else 0

            if show_slice_labels:
                ax.text(x, y + 0.08, label, ha='center', va='center',
                        fontsize=annot_fontsize, color=text_color)

            if pct >= min_pct_to_label:
                pct_str = f"{pct:.0f}%" if pct >= 1 else f"{pct:.1f}%"
                ax.text(x, y - 0.08, pct_str, ha='center', va='center',
                        fontsize=annot_fontsize, color=text_color)

        if show_subplot_titles:
            ax.set_title(title.title(), fontsize=title_fontsize, pad=title_pad)
        ax.set_aspect("equal")
        wedges_all.append(wedges)

    # legend
    from textwrap import wrap

    def wrap_label(s, width=text_width):
        s = str(s)
        prev = None
        while prev != s:
            prev = s
            s = re.sub(r"\s*\([^()]*\)", "", s)  # remove " ( ... ) " blocks
        # Tidy extra spaces created by removals
        s = re.sub(r"\s{2,}", " ", s).strip()
        # break_long_words=False keeps whole words when possible
        return "\n".join(wrap(s, width=width, break_long_words=False))

    if show_legend:
        legend_handles = [
            Line2D([0], [0], marker="o", linestyle="None",
                   markerfacecolor=color_palette[g], markeredgecolor="white",
                   markersize=15, label=wrap_label(g))
            for g in stat_order
        ]

        fig.legend(handles=legend_handles, loc="center left", frameon=False, fontsize=legend_fontsize,
                   bbox_to_anchor=(1.00, 0.5))

    if shared_title:
        fig.subplots_adjust(top=0.85)
        fig.suptitle(shared_title, fontsize=title_fontsize + 2, y=0.97)

    # save plot
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(size_inches_x, size_inches_y)
    plt.savefig(os.path.join(save_path, f"{save_name}.{fmt}"), format=fmt, dpi=1000, bbox_inches="tight", pad_inches=0.01)
    del figure
    plt.clf()
    plt.close()

    # save data to csv
    df[stat_col] = df[stat_col].astype("string").fillna("NaN")
    ct = (df.groupby([split_col, stat_col], dropna=False).size().reset_index(name="count"))
    result = ct.pivot(index=split_col, columns=stat_col, values="count").fillna(0).reset_index()
    result.to_csv(os.path.join(save_path, f"{save_name}.csv"), index=False)

    return


def _to_gray(c):
    # legend helper
    from matplotlib import colors as mcolors
    r, g, b = mcolors.to_rgb(c)
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b  # luminance
    return (y, y, y)


def plot_horizontal_stacked_props(df, save_path, save_name, split_col, split_order, rating_col, ratings_order,
                                  color_maps, show_pct_labels, label_min_pct=0, bar_height=0.6,
                                  legend=None, label_colors=None, size_inches_x=12, size_inches_y=8,
                                  fmt="svg", annotation_size=16, legend_fontsize=16, tick_fontsize=18, bar_gap=0.2):

    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    if split_order is None:
        split_order = list(pd.Index(df[split_col]).dropna().unique())
    df = df.copy()
    df[split_col] = pd.Categorical(df[split_col], categories=split_order, ordered=True)
    df[rating_col] = pd.Categorical(df[rating_col], categories=ratings_order, ordered=True)

    # Proportion table: rows = groups, cols = ratings (0..100)
    # first - counts
    counts = pd.crosstab(df[split_col], df[rating_col], dropna=False)
    counts = (counts.reindex(index=split_order, fill_value=0).reindex(columns=ratings_order, fill_value=0))
    counts.index.name = split_col
    counts.columns.name = rating_col
    # convert to proportions
    props = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) * 100.0

    # Create figure/axes sized by number of groups
    bar_spacing = bar_height + bar_gap
    h = max(2.5, bar_spacing * len(split_order) + 1.0)
    fig, ax = plt.subplots(figsize=(9, h))

    # Default colors if a specific mapping is missing
    default_palette = sns.color_palette(n_colors=len(ratings_order))
    default_color_by_rating = {r: default_palette[i % len(default_palette)]
                               for i, r in enumerate(ratings_order)}

    y_positions = np.arange(len(split_order)) * bar_spacing

    # Draw stacked bars
    for yi, grp in enumerate(split_order):
        y = y_positions[yi]
        left = 0.0
        for r in ratings_order:
            w = float(props.loc[grp, r])
            if w <= 0:
                continue
            # pick color: per-group map > default by rating
            color = None
            if color_maps is not None:
                cmap_g = color_maps.get(grp, {})
                color = cmap_g.get(r, None)
            if color is None:
                color = default_color_by_rating[r]

            ax.barh(y, w, left=left, height=bar_height, color=color, edgecolor="white")

            if show_pct_labels and w >= label_min_pct:
                ax.text(left + w / 2.0, y, f"{w:.0f}%", va="center", ha="center",
                        fontsize=annotation_size, color="black")
            left += w

    # axis
    ax.set_yticks(y_positions, labels=[str(g).title() for g in split_order], fontsize=tick_fontsize)  # names of ticks of split
    ax.set_ylim(-bar_spacing * 0.5, y_positions[-1] + bar_spacing * 0.5)
    ax.set_xlim(0, 100)
    ax.tick_params(axis="x", labelsize=tick_fontsize)
    ax.set_xlabel("Proportion", fontsize=tick_fontsize + 2)
    ax.set_title("")
    ax.grid(axis="y", visible=False)

    # Legend
    if legend is not None:
        # pick colors for legend labels
        if label_colors is not None:
            legend_colors = {r: label_colors[r] for r in ratings_order}
        else:
            # derive from the first group that has a mapping; then grayscale it
            base_group = next((g for g in split_order if color_maps and g in color_maps), None)
            if base_group is not None:
                legend_colors = {r: _to_gray(color_maps[base_group].get(r, "0.5")) for r in ratings_order}
            else:
                # fallback to grayscale of default palette by rating
                legend_colors = {r: _to_gray(default_color_by_rating[r]) for r in ratings_order}

        # build circular handles (no rectangles)
        order_for_legend = ratings_order if legend == "inside" else list(ratings_order)[::-1]
        handles = [
            Line2D([0], [0],
                   marker='o', linestyle='None',
                   markerfacecolor=legend_colors[r],
                   markeredgecolor='white', markeredgewidth=0.8,
                   markersize=legend_fontsize, label=str(r))
            for r in order_for_legend
        ]

        # place legend
        loc = "upper center" if legend == "inside" else "center left"
        bba = (0.42, 1.07) if legend == "inside" else (1.0, 0.5)
        ax.legend(handles=handles, title="", loc=loc, bbox_to_anchor=bba, fontsize=legend_fontsize,
                  ncol=len(ratings_order) if legend == "inside" else 1, frameon=False,
                  borderaxespad=1.0,  # space between legend and axes
                  handletextpad=0.4,  # marker–label spacing
                  labelspacing=0.3,  # vertical spacing between entries
                  columnspacing=4.7)
    plt.tight_layout()

    # save plot
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(size_inches_x, size_inches_y)
    plt.savefig(os.path.join(save_path, f"{save_name}.{fmt}"), format=fmt, dpi=1000, bbox_inches="tight",
                pad_inches=0.01)
    del figure
    plt.clf()
    plt.close()

    # save data to csv
    # map to ensure numeric to add stats to the 'counts' df
    df_num = df.copy()
    if pd.api.types.is_numeric_dtype(df[rating_col]):
        df_num["_rating_num"] = df[rating_col]
    else:
        # Map categories to 1..N in the order of ratings_order
        mapping = {v: i for i, v in enumerate(ratings_order, start=1)}
        # Try a direct numeric conversion; if it fails, fall back to mapping
        tmp = pd.to_numeric(df[rating_col], errors="coerce")
        if tmp.notna().any() and tmp.isna().sum() < len(tmp):
            df_num["_rating_num"] = tmp
        else:
            df_num["_rating_num"] = df[rating_col].map(mapping).astype("float")
    stats = (df_num
             .groupby(split_col, dropna=False)["_rating_num"]
             .agg(mean="mean", sd=lambda s: s.std(ddof=1)))
    stats = stats.reindex(split_order)
    counts = counts.join(stats)
    counts.to_csv(os.path.join(save_path, f"{save_name}.csv"), index=False)
    return


def plot_higher_education(sub_df, save_path):
    x_col = "In what topic?"
    y_col = "sample"

    df_exp = sub_df.loc[:, ["response_id", x_col, y_col]].copy()
    df_exp = df_exp[df_exp[x_col].notnull()]

    def parse_multiselect(val):
        """Parse comma-separated values."""
        if isinstance(val, (list, tuple, set)):
            return [str(v).strip() for v in val if str(v).strip()]
        val_str = str(val)
        items = re.split(r"\s*,\s*", val_str)
        return [item.strip() for item in items if item.strip()]

    df_exp = (df_exp
              .assign(**{x_col: lambda d: d[x_col].apply(parse_multiselect)})
              .explode(x_col)
              .loc[lambda d: d[x_col].notna() & d[x_col].ne("")])

    # numerator: unique people per (sample, topic)
    ppl_per_topic = (df_exp[["response_id", y_col, x_col]]
                     .drop_duplicates()
                     .groupby([y_col, x_col], dropna=False)
                     .size()
                     .reset_index(name="n_people"))

    # denominator: unique people per sample who answered this question
    denom = (df_exp
             .groupby(y_col)["response_id"]
             .nunique()
             .rename("n_people_total")
             .reset_index())

    # merge & compute percent-of-people
    counts = ppl_per_topic.merge(denom, on=y_col, how="left")
    counts["percent"] = 100 * counts["n_people"] / counts["n_people_total"]

    # pick top N topics per sample by number of people
    top_n = 10
    top_per_y = (counts
                 .sort_values([y_col, "n_people"], ascending=[True, False])
                 .groupby(y_col, as_index=False, sort=False)
                 .head(top_n))
    keep_x = set(top_per_y[x_col])
    df_kept = counts[counts[x_col].isin(keep_x)].copy()

    plot_hist_multiselect(df=df_kept, save_path=save_path, save_name="hist_eduTopic", fmt="svg",
                          x_col=x_col, y_col=y_col,
                          y_order=["exploratory", "pre-registered", "follow-up"],
                          title="Higher Education Topics",
                          y_title="Proportion (%)", x_title="",
                          n_col="n_people", denom_col="n_people_total",
                          metric="percent",
                          size_inches_x=16, size_inches_y=8,
                          alpha_val=1.0,
                          color_palette=SAMPLE_COLORS,
                          x_ticks=None,
                          y_ticks=np.arange(0, 26, 5),
                          x_tick_rotation=30,
                          bar_shrink=0.8,
                          x_text_width=22,
                          title_fontsize=28,
                          y_tick_fontsize=22,
                          x_tick_fontsize=20,
                          ylabel_fontsize=24,
                          xlabel_fontsize=24,
                          legend_fontsize=20)

    df_kept.to_csv(os.path.join(save_path, "hist_eduTopic_data.csv"), index=False)
    return


def plot_employment(sub_df, save_path):
    x_col = survey_mapping.Q_EMPLOYMENT
    y_col = "sample"

    df_exp = sub_df.loc[:, ["response_id", x_col, y_col]].copy()
    df_exp = df_exp[df_exp[x_col].notnull()]

    # numerator: unique people per (sample, category)
    ppl_per_topic = (df_exp[["response_id", y_col, x_col]]
                     .drop_duplicates()
                     .groupby([y_col, x_col], dropna=False)
                     .size()
                     .reset_index(name="n_people"))

    # denominator: unique people per sample
    denom = (df_exp
             .groupby(y_col)["response_id"]
             .nunique()
             .rename("n_people_total")
             .reset_index())

    # merge & compute percent-of-people
    counts = ppl_per_topic.merge(denom, on=y_col, how="left")
    counts["percent"] = 100 * counts["n_people"] / counts["n_people_total"]

    # pick top N categories per sample
    top_n = 10
    top_per_y = (counts
                 .sort_values([y_col, "n_people"], ascending=[True, False])
                 .groupby(y_col, as_index=False, sort=False)
                 .head(top_n))
    keep_x = set(top_per_y[x_col])
    df_kept = counts[counts[x_col].isin(keep_x)].copy()

    plot_hist_multiselect(df=df_kept, save_path=save_path, save_name="hist_employment", fmt="svg",
                          x_col=x_col, y_col=y_col,
                          y_order=["exploratory", "pre-registered", "follow-up"],
                          title="Employment Domain",
                          y_title="Proportion (%)", x_title="",
                          n_col="n_people", denom_col="n_people_total",
                          metric="percent",
                          size_inches_x=20, size_inches_y=8,
                          alpha_val=1.0,
                          color_palette=SAMPLE_COLORS,
                          x_ticks=None,
                          y_ticks=np.arange(0, 31, 5),
                          x_tick_rotation=30,
                          bar_shrink=0.8,
                          x_text_width=24,
                          title_fontsize=28,
                          y_tick_fontsize=22,
                          x_tick_fontsize=20,
                          ylabel_fontsize=24,
                          xlabel_fontsize=24,
                          legend_fontsize=20)

    df_kept.to_csv(os.path.join(save_path, "hist_employment_data.csv"), index=False)
    return


def plot_demographics(df, save_path):

    # Age
    plot_hist(df=df, save_path=save_path, save_name="hist_age", fmt="svg",
              stat_type="percent",  # not counts
              element="poly",  # no bars (hist line)
              multiple_resolve="layer",  # overlay groups (overlap, semi-transparent)
              x_col="How old are you?", x_title="Age",
              y_col="sample",  y_order=["pre-registered", "exploratory", "follow-up"],
              title="",  y_title="Proportion (%)",
              size_inches_x=12, size_inches_y=8, alpha_val=0.35, color_palette=SAMPLE_COLORS,
              bins=np.arange(18, 81, 2, dtype=float), x_ticks=np.arange(20, 81, 10), y_ticks=np.arange(0, 16, 5))

    # Gender
    plot_pie(df=df, save_path=save_path, save_name="pie_gender", fmt="svg",
             stat_col="How do you describe yourself?",
             split_col="sample", split_order=["exploratory", "pre-registered", "follow-up"],
             stat_order=["Female", "Male", "Non-binary", "Genderqueer", "Genderfluid", "Prefer not to say"],
             color_palette=GENDER_COLORS,
             size_inches_x=12, size_inches_y=6,
             annot_fontsize=26,
             title_fontsize=24,
             legend_fontsize=22,
             shared_title="Gender")

    # Country
    analyze_survey.demographics_country(demographics_df=df, save_path=save_path)

    # Education
    plot_pie(df=df, save_path=save_path, save_name="pie_edu", fmt="svg",
             stat_col="What is your education background?",
             split_col="sample", split_order=["exploratory", "pre-registered", "follow-up"],
             stat_order=survey_mapping.EDU_ORDER,
             color_palette=EDU_COLORS,
             size_inches_x=12, size_inches_y=6,
             annot_fontsize=26,
             title_fontsize=24,
             legend_fontsize=22,
             shared_title="Education Background")

    # Higher education - topic: present only the top-10 topics for each category
    plot_higher_education(df, save_path)

    # Current employment domain
    plot_employment(df, save_path)

    return


def plot_experience(sub_df, save_path):
    experience = {"experienceAnimals": survey_mapping.Q_ANIMAL_EXP,
                  "experienceAI": survey_mapping.Q_AI_EXP,
                  "experienceCons": survey_mapping.Q_CONSC_EXP,
                  "experienceEthics": survey_mapping.Q_ETHICS_EXP}

    exp_titles = {"experienceAnimals": "Animal Experience",
                  "experienceAI": "AI Experience",
                  "experienceCons": "Consciousness Experience",
                  "experienceEthics": "Ethics Experience"}

    for exp_name in experience:  # iterate over keys
        exp_col = experience[exp_name]
        exp_title = exp_titles[exp_name]

        relevant_df = sub_df.loc[:, ["response_id", "sample", exp_col]].copy()
        relevant_df = relevant_df.rename(columns={exp_col: "Rating"})
        relevant_df["Experience"] = exp_title

        plot_hist_separate_groups(df=relevant_df, save_path=save_path, save_name=f"hist_{exp_name}",
                                  y_order=[exp_title], y_col="Experience",
                                  facet="item",
                                  split_col="sample", split_order=["follow-up", "pre-registered", "exploratory"],
                                  rating_col="Rating", ratings_order=[1, 2, 3, 4, 5],
                                  bar_height=0.5, bar_pitch=0.6,
                                  color_maps=SAMPLE_COLORS_RATINGS,
                                  show_pct_labels=True,
                                  subplot_hspace=0.15,
                                  fmt="svg",
                                  size_inches_x=12, size_inches_y=3,
                                  legend="top",
                                  legend_fontsize=20,
                                  annot_fontsize=22,
                                  title_fontsize=25,
                                  y_tick_fontsize=24,
                                  show_y_tick_labels=True,
                                  show_group_legend=False,
                                  legend_reverse=False,
                                  min_pct_label=0,
                                  shared_title=exp_title,
                                  shared_title_y=1.25,
                                  x_label="Proportion (%)", x_label_fontsize=20)
    return


def plot_experience_followup(sub_df, save_path):

    # pets
    plot_pie(df=sub_df, save_path=save_path, save_name="pie_pets", fmt="svg",
             stat_col=survey_mapping.Q_PETS,
             split_col="sample", split_order=["exploratory", "pre-registered", "follow-up"],
             stat_order=["Yes", "No"],
             color_palette=YESNO_COLORS,
             size_inches_x=12, size_inches_y=6,
             annot_fontsize=26,
             title_fontsize=24,
             legend_fontsize=22, shared_title=survey_mapping.Q_PETS.title())

    # experience follow up - 3 and up
    experience = {"expeAnimals": [survey_mapping.Q_ANIMAL_EXP_FOLLOW_UP, np.arange(0, 101, 10)],
                  "expeAI": [survey_mapping.Q_AI_EXP_FOLLOW_UP, np.arange(0, 101, 10)],
                  "expeCons": [survey_mapping.Q_CONSC_EXP_FOLLOW_UP, np.arange(0, 101, 10)],
                  "expeEthics": [survey_mapping.Q_ETHICS_EXP_FOLLOW_UP, np.arange(0, 71, 10)]}

    exp_titles = {
        "expeAnimals": "Animal Experience",
        "expeAI": "AI Experience",
        "expeCons": "Consciousness Experience",
        "expeEthics": "Ethics Experience"
    }

    PAREN = r"\s*\([^()]*\)"  # pattern to remove " ( ... )" including space before

    for exp_name in experience:
        x_col = experience[exp_name][0]
        y_ticks = experience[exp_name][1]
        y_col = "sample"

        df_exp = sub_df.loc[:, ["response_id", x_col, y_col]].copy()
        df_exp = df_exp[df_exp[x_col].notnull()]  # only experience 3+ answered this

        def parse_multiselect(val):
            """Parse comma-separated values, removing parenthetical content first."""
            if isinstance(val, (list, tuple, set)):
                return [re.sub(PAREN, "", str(v)).strip() for v in val]
            val_str = str(val)
            # Remove parenthetical content first, then split by comma
            val_clean = re.sub(PAREN, "", val_str)
            items = re.split(r"\s*,\s*", val_clean)
            return [item.strip() for item in items if item.strip()]

        df_exp = (df_exp
        .assign(**{x_col: lambda d: d[x_col].apply(parse_multiselect)})
        .explode(x_col)
        .loc[lambda d: d[x_col].notna() & d[x_col].ne("")])

        # numerator: count unique people per (sample, topic)
        ppl_per_topic = (df_exp[["response_id", y_col, x_col]]
                         .drop_duplicates()
                         .groupby([y_col, x_col], dropna=False)
                         .size()
                         .reset_index(name="n_people"))

        # denominator: unique people per sample who answered this question
        denom = (df_exp
                 .groupby(y_col)["response_id"]
                 .nunique()
                 .rename("n_people_total")
                 .reset_index())

        # merge & compute percent-of-people
        counts = ppl_per_topic.merge(denom, on=y_col, how="left")
        counts["percent"] = 100 * counts["n_people"] / counts["n_people_total"]

        # pick top N topics per sample by number of people
        top_n = 10
        top_per_y = (counts
                     .sort_values([y_col, "n_people"], ascending=[True, False])
                     .groupby(y_col, as_index=False, sort=False)
                     .head(top_n))
        keep_x = set(top_per_y[x_col])
        df_kept = counts[counts[x_col].isin(keep_x)].copy()

        plot_hist_multiselect(df=df_kept, save_path=save_path, save_name=f"hist_{exp_name}_extra", fmt="svg",
                              x_col=x_col, y_col=y_col,
                              y_order=["exploratory", "pre-registered", "follow-up"],
                              title=exp_titles[exp_name],
                              y_title="Proportion (%)", x_title="",
                              n_col="n_people", denom_col="n_people_total",
                              metric="percent",
                              size_inches_x=16, size_inches_y=8,
                              alpha_val=1.0,
                              color_palette=SAMPLE_COLORS,
                              x_ticks=None,
                              y_ticks=y_ticks,
                              x_tick_rotation=30,
                              bar_shrink=0.8,
                              x_text_width=22,
                              title_fontsize=28,
                              y_tick_fontsize=22,
                              x_tick_fontsize=20,
                              ylabel_fontsize=24,
                              xlabel_fontsize=24,
                              legend_fontsize=20)

        df_kept.to_csv(os.path.join(save_path, f"hist_{exp_name}_data.csv"), index=False)

    return


def dedup_annotation_indices(df, x_col, y_col, identity_col, eps=0.08):
    """
    HELPER ANNOTATION FUNC
    Return a boolean mask of rows to KEEP for annotation.
    For each unique identity (the text label), nearby points within `eps`
    are clustered and only one label per cluster is kept.
    """
    keep = np.zeros(len(df), dtype=bool)

    for label, grp in df.groupby(identity_col):
        pts = grp[[x_col, y_col]].to_numpy()
        grp_idx = grp.index.to_numpy()

        if len(grp_idx) == 1:
            keep[grp_idx[0]] = True
            continue

        # Try fast path with cKDTree
        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(pts)
            pairs = tree.query_pairs(r=eps)  # set of (i, j) pairs within eps

            # Union-Find over points in `pts`
            parent = np.arange(len(pts))
            def find(a):
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a
            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

            for a, b in pairs:
                union(a, b)

            # Keep the first point encountered from each connected component
            seen = set()
            for i in range(len(pts)):
                r = find(i)
                if r not in seen:
                    seen.add(r)
                    keep[grp_idx[i]] = True

        except Exception:
            # Fallback: greedy O(n^2) dedup
            taken = np.zeros(len(pts), dtype=bool)
            for i in range(len(pts)):
                if taken[i]:
                    continue
                keep[grp_idx[i]] = True
                d = np.sqrt(((pts - pts[i]) ** 2).sum(axis=1))
                taken |= (d <= eps)

    return keep


def tint_image(img_array, color, border_width=0, border_color=(80, 80, 80)):
    """
    Tint an RGBA image with a given color.
    Preserves alpha channel, replaces RGB with the target color.
    """
    rgb = tuple(int(c * 255) for c in to_rgb(color))
    tinted = img_array.copy()

    if border_width > 0:
        # Create mask of non-transparent pixels
        alpha_mask = img_array[:, :, 3] > 0

        # Use distance transform for smoother edges
        dist = distance_transform_edt(alpha_mask)

        # Inner region is where distance > border_width
        inner_mask = dist > border_width

        # Border is non-transparent but not inner
        border_mask = alpha_mask & ~inner_mask

        # Tint inner region with target color
        tinted[:, :, 0] = np.where(inner_mask, rgb[0], tinted[:, :, 0])
        tinted[:, :, 1] = np.where(inner_mask, rgb[1], tinted[:, :, 1])
        tinted[:, :, 2] = np.where(inner_mask, rgb[2], tinted[:, :, 2])

        # Set border to border_color
        tinted[:, :, 0] = np.where(border_mask, border_color[0], tinted[:, :, 0])
        tinted[:, :, 1] = np.where(border_mask, border_color[1], tinted[:, :, 1])
        tinted[:, :, 2] = np.where(border_mask, border_color[2], tinted[:, :, 2])
    else:
        # No border - tint entire image
        tinted[:, :, 0] = rgb[0]
        tinted[:, :, 1] = rgb[1]
        tinted[:, :, 2] = rgb[2]

    return tinted


def load_and_normalize_icon(path, target_height, max_width=None):
    """
    Load an icon and normalize to target height, with optional max width constraint.
    """
    img = Image.open(path).convert("RGBA")
    w, h = img.size

    # First scale to target height
    new_height = target_height
    new_width = int(w * (new_height / h))

    # If too wide, scale down to max_width instead
    if max_width is not None and new_width > max_width:
        new_width = max_width
        new_height = int(h * (new_width / w))

    img_resized = img.resize((new_width, new_height), Image.LANCZOS)
    return np.array(img_resized)


def plot_scatter_icons(df, x_col, y_col, identity_col, group_col, group_order, group_colors,
                       icon_paths, save_path, save_name,
                       x_range, y_range, icon_size=50, icon_max_width=None, alpha=0.85,
                       vertical_jitter=0, horizontal_jitter=0,
                       size_inches_x=18, size_inches_y=12, fmt="svg",
                       legend_fontsize=20, diag_line=True,
                       show_annotations=True, show_icon_legend=True, show_group_legend=True,
                       icon_legend_size=30, icon_legend_max_width=None,
                       icon_legend_fontsize=14, icon_legend_spacing=1.0,
                       plot_left=0.08, plot_width=0.7, legend_left=0.82, legend_width=0.15,
                       border_width=0, border_color=(50, 50, 50),
                       x_tick_labels=None, y_tick_labels=None, y_tick_rotation=0,
                       axis_label_fontsize=22, tick_fontsize=18):
    """
    Scatter plot using colored PNG icons instead of circles.

    Parameters
    ----------
    icon_size : int
        Target height in pixels for icons.
    icon_max_width : int or None
        Maximum width in pixels for icons. If None, no constraint.
    icon_legend_spacing : float
        Vertical spacing multiplier for legend items (1.0 = default, >1 = more space).
    plot_left, plot_width, legend_left, legend_width : float
        Control positioning of plot and legend (as fractions of figure).
    """
    df = df.copy()

    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    fig, ax = plt.subplots(figsize=(size_inches_x, size_inches_y))

    # Set axis limits first
    x_min, x_max = min(x_range) - 0.1, max(x_range) + 0.25
    y_min, y_max = min(y_range) - 0.1, max(y_range) + 0.25
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # diagonal dashed line
    if diag_line:
        start = min(x_range)
        end = max(x_range)
        ax.plot([start, end], [start, end], color="#1F2041", linestyle="dashed",
                linewidth=1.25, zorder=2)

    # jitter
    counts = df.groupby([x_col, y_col]).size().reset_index(name="counts")
    jitter_mask = counts[counts["counts"] > 1].set_index([x_col, y_col]).index

    if vertical_jitter > 0 or horizontal_jitter > 0:
        for i in range(len(df)):
            if (df[x_col].iat[i], df[y_col].iat[i]) in jitter_mask:
                if vertical_jitter > 0:
                    df[y_col].iat[i] += np.random.uniform(-vertical_jitter, vertical_jitter)
                if horizontal_jitter > 0:
                    df[x_col].iat[i] += np.random.uniform(-horizontal_jitter, horizontal_jitter)

    # load and normalize icons
    icon_scale = 8  # render at 4x, then zoom down
    icon_cache = {}
    for identity, path in icon_paths.items():
        if os.path.exists(path):
            icon_cache[identity] = load_and_normalize_icon(path, icon_size * icon_scale,
                                                           icon_max_width * icon_scale if icon_max_width else None)

    # plot icons
    for i in range(len(df)):
        x = df[x_col].iat[i]
        y = df[y_col].iat[i]
        identity = df[identity_col].iat[i]
        group = df[group_col].iat[i]

        if identity in icon_cache and group in group_colors:
            base_img = icon_cache[identity]
            tinted_img = tint_image(base_img, group_colors[group],
                                    border_width=int(border_width * icon_scale),
                                    border_color=border_color)
            tinted_img = tinted_img.copy()
            tinted_img[:, :, 3] = (tinted_img[:, :, 3] * alpha).astype(np.uint8)

            imagebox = OffsetImage(tinted_img, zoom=1.0 / icon_scale)
            imagebox.image.axes = ax
            ab = AnnotationBbox(imagebox, (x, y), frameon=False, zorder=3)
            ax.add_artist(ab)

    # axis labels and ticks
        # axis labels and ticks
        ax.set_xlabel(x_col, fontsize=axis_label_fontsize, labelpad=10)
        ax.set_ylabel(y_col, fontsize=axis_label_fontsize, labelpad=8)
        ax.set_xticks(x_range)
        ax.set_yticks(y_range)

        # custom tick labels if provided
        if x_tick_labels:
            labels = [x_tick_labels.get(t, str(t)) for t in x_range]
            ax.set_xticklabels(labels, fontsize=tick_fontsize)
        else:
            ax.tick_params(axis='x', labelsize=tick_fontsize)

        if y_tick_labels:
            labels = [y_tick_labels.get(t, str(t)) for t in y_range]
            ax.set_yticklabels(labels, fontsize=tick_fontsize, rotation=y_tick_rotation, va='center', ha='right')
        else:
            ax.tick_params(axis='y', labelsize=tick_fontsize)

    # re-enforce limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # annotations
    if show_annotations:
        eps = 0.05
        keep_mask = dedup_annotation_indices(df, x_col, y_col, identity_col, eps=eps)
        kept_indices = np.flatnonzero(keep_mask)
        for i in kept_indices:
            ax.text(df[x_col].iat[i], df[y_col].iat[i] + 0.15,
                    df[identity_col].iat[i], fontsize=18, ha="center")

    # group color legend (colored circles)
    if show_group_legend:
        handles = [
            Line2D([0], [0],
                   marker='o', linestyle='',
                   markersize=legend_fontsize,
                   markerfacecolor=group_colors[g],
                   markeredgecolor='none')
            for g in group_order
        ]
        labels = [str(g).title() for g in group_order]
        group_legend = ax.legend(handles, labels, title="", frameon=False,
                                 loc="upper left", fontsize=legend_fontsize)
        ax.add_artist(group_legend)

    # icon legend
    if show_icon_legend:
        legend_scale = 8
        legend_icon_cache = {}
        for identity, path in icon_paths.items():
            if os.path.exists(path):
                legend_icon_cache[identity] = load_and_normalize_icon(path, icon_legend_size * legend_scale,
                                                                      icon_legend_max_width * legend_scale if icon_legend_max_width else None)

        unique_identities = sorted(df[identity_col].unique())  # alphabetically ordered

        legend_ax = fig.add_axes([legend_left, 0.15, legend_width, 0.77])
        legend_ax.axis('off')

        n_items = len(unique_identities)
        # adjust spacing - use spacing parameter to control vertical distribution
        top = 0.98
        bottom = 1.0 - (icon_legend_spacing * 0.95)
        bottom = max(0.02, bottom)
        y_positions = np.linspace(top, bottom, n_items)

        for identity, y_pos in zip(unique_identities, y_positions):
            if identity in legend_icon_cache:
                img_array = legend_icon_cache[identity].copy()
                img_array[:, :, 0] = 80
                img_array[:, :, 1] = 80
                img_array[:, :, 2] = 80

                imagebox = OffsetImage(img_array, zoom=1.0 / legend_scale)  # zoom down
                ab = AnnotationBbox(imagebox, (0.1, y_pos), frameon=False,
                                    xycoords='axes fraction', box_alignment=(0.5, 0.5))
                legend_ax.add_artist(ab)

                legend_ax.text(0.25, y_pos, identity, fontsize=icon_legend_fontsize,
                               va='center', ha='left', transform=legend_ax.transAxes)

    # adjust main plot position
    if show_icon_legend:
        ax.set_position([plot_left, 0.12, plot_width, 0.82])

    plt.savefig(os.path.join(save_path, f"{save_name}.{fmt}"), format=fmt, dpi=1000,
                bbox_inches="tight", pad_inches=0.01)
    plt.clf()
    plt.close()

    return


def plot_scatter(df, x_col, y_col, identity_col, group_col, group_order, group_colors, save_path, save_name,
                 x_range, y_range, size, alpha, vertical_jitter=0, horizontal_jitter=0,
                 size_inches_x=18, size_inches_y=12, fmt="svg", legend_fontsize=20, diag_line=True):

    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    # diagonal dashed line
    if diag_line is True:
        start = math.floor(min(df[x_col].tolist()))
        end = math.ceil(max(df[x_col].tolist()))
        # draw a line from (start, start) to (end, end)
        plt.plot([start, end], [start, end], color="#1F2041", linestyle="dashed", linewidth=1.25, zorder=2)

    # jitter
    # count duplicates for (x, y) pairs - there's no need to jitter if it's the only dot with these (x, y) values
    counts = df.groupby([x_col, y_col]).size().reset_index(name="counts")

    # create a mask for points that have duplicates
    jitter_mask = counts[counts["counts"] > 1].set_index([x_col, y_col]).index

    # jitter accordingly
    if vertical_jitter > 0 or horizontal_jitter > 0:
        for i in range(len(df)):
            if (df[x_col][i], df[y_col][i]) in jitter_mask:
                if vertical_jitter > 0:
                    df[y_col].iat[i] += np.random.uniform(-vertical_jitter, vertical_jitter)
                if horizontal_jitter > 0:
                    df[x_col].iat[i] += np.random.uniform(-horizontal_jitter, horizontal_jitter)

    ax = sns.scatterplot(data=df, x=x_col, y=y_col, hue=group_col, palette=group_colors, hue_order=group_order,
                         s=size, alpha=alpha, zorder=3, linewidth=0, legend="brief")

    # axis
    plt.xlabel(x_col, fontsize=25, labelpad=10)
    plt.ylabel(y_col, fontsize=25, labelpad=8)
    plt.yticks(y_range, fontsize=22)
    plt.xticks(x_range, fontsize=22)

    # annotate
    eps = 0.05  # the limit beyond which a duplicate annotation (same identity) is too close
    keep_mask = dedup_annotation_indices(df, x_col, y_col, identity_col, eps=eps)
    kept_indices = np.flatnonzero(keep_mask)
    # for i in range(len(df)):
    # plt.text(df[x_col][i], df[y_col][i] + 0.065, df[identity_col][i], fontsize=20, ha="center")
    for i in kept_indices:
        plt.text(df[x_col].iat[i], df[y_col].iat[i] + 0.025, df[identity_col].iat[i], fontsize=18, ha="center")

    # legend
    handles = [
        Line2D([0], [0],
               marker='o', linestyle='',
               markersize=legend_fontsize,
               markerfacecolor=group_colors[g],
               markeredgecolor='none')
        for g in group_order
    ]
    labels = [str(g).title() for g in group_order]
    ax.legend(handles, labels, title="", frameon=False, loc="best", fontsize=legend_fontsize)
    plt.tight_layout()

    # save plot
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(size_inches_x, size_inches_y)
    plt.savefig(os.path.join(save_path, f"{save_name}.{fmt}"), format=fmt, dpi=1000, bbox_inches="tight",
                pad_inches=0.01)
    del figure
    plt.clf()
    plt.close()

    return


def plot_c_v_ms(sub_df, save_path):
    ms_cols = list(survey_mapping.other_creatures_ms.values())
    c_cols = list(survey_mapping.other_creatures_cons.values())
    corr_df = sub_df.loc[:, ["response_id", "sample"] + ms_cols + c_cols]

    # melt
    long_data = pd.melt(corr_df, id_vars=["sample", "response_id"], var_name="Item_Topic", value_name="Rating")
    long_data[["Topic", "Item"]] = long_data["Item_Topic"].str.split('_', expand=True)
    long_data = long_data.drop(columns=["Item_Topic"])
    long_data["Topic"] = long_data["Topic"].map({"c": "Consciousness", "ms": "Moral Status"})
    # pivot
    df_pivot = long_data.pivot_table(index=["sample", "Item"], columns="Topic", values="Rating",
                                     aggfunc="mean").reset_index(drop=False, inplace=False)
    df_pivot["Item"] = df_pivot["Item"].replace(survey_mapping.other_creatures_general_names)

    short_name_to_icon = {
        survey_mapping.other_creatures_general_names[k]: v
        for k, v in survey_mapping.other_creatures_icon_paths.items()
    }

    plot_scatter_icons(
        df=df_pivot, x_col="Consciousness", y_col="Moral Status", identity_col="Item",
        group_col="sample", group_order=["exploratory", "pre-registered", "follow-up"],
        group_colors=SAMPLE_COLORS, icon_paths=short_name_to_icon,
        x_range=np.arange(1, 5, 1), y_range=np.arange(1, 5, 1),
        icon_size=30,
        icon_max_width=30,  # constrain wide icons
        alpha=0.80,
        border_width=0.7,  # icon border thickness in pixels: thinnest visible border would be 0.25
        border_color=(40, 45, 50),  # if we have a width, border color
        vertical_jitter=0, horizontal_jitter=0,  # add jitter to avoid overlaps
        size_inches_x=18,  # wider figure
        size_inches_y=10,
        fmt="svg",
        save_path=save_path, save_name="scatter_cVms_icons",
        legend_fontsize=18,
        diag_line=True,
        show_annotations=False,
        show_icon_legend=True,
        icon_legend_size=25,
        icon_legend_max_width=25,  # constrain legend icons too
        icon_legend_fontsize=13,
        icon_legend_spacing=1.4,  # adjust if needed (>1 = more space)
        plot_left=0.06,  # move plot left edge
        plot_width=0.78,  # wider plot area
        legend_left=0.79,  # icon legend to the right --> 1
        legend_width=0.20,  # legend width
    )

    #plot_scatter(df=df_pivot, x_col="Consciousness", y_col="Moral Status", identity_col="Item", group_col="sample",
    #             group_order=["exploratory", "pre-registered", "follow-up"], group_colors=SAMPLE_COLORS,
    #             x_range=np.arange(1, 5, 1), y_range=np.arange(1, 5, 1), size=600, alpha=0.85,
    #             vertical_jitter=0, horizontal_jitter=0, size_inches_x=18, size_inches_y=12, fmt="svg",
    #             save_path=save_path, save_name="scatter_cVms", legend_fontsize=20, diag_line=True)
    return


def plot_hist_separate_groups(df, save_path, save_name, y_order, y_col, split_col, split_order,
                              rating_col, ratings_order, bar_height, color_maps, show_pct_labels,
                              fmt="svg", size_inches_x=15, size_inches_y=10, legend=None, legend_fontsize=18,
                              facet="split", subplot_hspace=0.35, bar_pitch=1.0,
                              show_y_tick_labels=True, show_group_legend=False,
                              annot_fontsize=12, title_fontsize=15, legend_reverse=False,
                              multi_col_threshold=10, subplot_wspace=0.25, min_pct_label=0,
                              x_label="", x_label_fontsize=None, y_tick_fontsize=None, show_x_per_subplot=False,
                              shared_title="", shared_title_y=1.08):
    counts = (
        df.groupby([split_col, y_col, rating_col], dropna=False)
        .size()
        .unstack(rating_col, fill_value=0)
        .reindex(columns=ratings_order, fill_value=0)
    )

    # ensure every  (split, item) exists even if absent in data
    idx = pd.MultiIndex.from_product([split_order, y_order], names=[split_col, y_col])
    counts = counts.reindex(idx, fill_value=0)
    # Row-normalize to proportions within each (split, item)
    props = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).mul(100).fillna(0)

    # color lookup
    # find a base per-rating map
    if color_maps and isinstance(next(iter(color_maps.values())), dict):
        # nested: {split: {rating: color}}
        # choose the first split that has a dict
        base_group = next((g for g in split_order if g in color_maps and isinstance(color_maps[g], dict)), None)
        base_color_map = color_maps.get(base_group, {})
    else:
        # flat: {rating: color}
        base_color_map = color_maps or {}

    def get_color_for_rating(r, split_val=None):
        # try split-specific first if provided, fall back to base per-rating
        if split_val is not None and isinstance(color_maps.get(split_val, None), dict):
            return color_maps[split_val].get(r, base_color_map.get(r, None))
        return base_color_map.get(r, None)

    if y_tick_fontsize is None:
        y_tick_fontsize = annot_fontsize

    # Determine grid layout
    if facet == "split":
        n_items = len(split_order)
        item_list = split_order
    else:  # facet == "item"
        n_items = len(y_order)
        item_list = y_order

    # Decide on single vs two-column layout
    if n_items > multi_col_threshold:
        ncols = 2
        nrows = (n_items + 1) // 2  # ceiling division
    else:
        ncols = 1
        nrows = n_items

    # Create figure with appropriate grid
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(size_inches_x, size_inches_y), sharex=True,
        gridspec_kw=dict(hspace=subplot_hspace, wspace=subplot_wspace, top=0.95, left=0.08, right=0.98, bottom=0.05)
    )

    # Flatten axes for easy iteration, handle edge cases
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    # Create flat list of axes in column-major order (fill left column first, then right)
    axes_flat = []
    for row in range(nrows):
        for col in range(ncols):
            axes_flat.append(axes[row, col])

    # Reorder to fill left column first, then right column
    if ncols == 2:
        axes_ordered = []
        for col in range(ncols):
            for row in range(nrows):
                axes_ordered.append(axes[row, col])
        axes_flat = axes_ordered

    # plot
    if facet == "split":
        for idx, (ax, split_val) in enumerate(zip(axes_flat[:n_items], split_order)):
            sub = props.loc[split_val].reindex(y_order)
            y_pos = np.arange(len(y_order)) * bar_pitch
            ax.set_ylim(y_pos[0] - bar_height / 2, y_pos[-1] + bar_height / 2)
            left = np.zeros(len(y_order))

            for r in ratings_order:
                w = sub[r].values
                seg_color = color_maps.get(split_val, {}).get(r, None)
                bars = ax.barh(y_pos, w, left=left, color=seg_color, edgecolor='none', height=bar_height)

                if show_pct_labels:
                    text_color = get_text_color(seg_color)
                    for rect, val, start in zip(bars.patches, w, left):
                        if val < min_pct_label:  # skip small values
                            continue
                        x_text = start + val / 2.0
                        ax.text(x_text, rect.get_y() + rect.get_height() / 2.0,
                                f"{val:.0f}%", va='center', ha='center', fontsize=annot_fontsize, color=text_color)
                left += w

            # Axes cosmetics
            ax.set_yticks(y_pos)
            ax.set_yticklabels(y_order)
            ax.tick_params(axis='y', labelsize=y_tick_fontsize)
            if not show_y_tick_labels:
                ax.set_yticklabels([])
                ax.tick_params(axis='y', length=0)
            if not shared_title:
                ax.set_title(f"{split_val.title()}", fontsize=title_fontsize)
            ax.grid(axis='x', linestyle=':', alpha=0.5)

    else:  # facet "item"
        for idx, (ax, item_val) in enumerate(zip(axes_flat[:n_items], y_order)):
            sub = (
                props.xs(item_val, level=y_col)
                .reindex(index=split_order)
            )

            y_pos = np.arange(len(split_order)) * bar_pitch
            ax.set_ylim(y_pos[0] - bar_height / 2, y_pos[-1] + bar_height / 2)
            left = np.zeros(len(split_order))

            for r in ratings_order:
                w = sub[r].values
                colors = [get_color_for_rating(r, split_val=s) for s in split_order]
                bars = ax.barh(y_pos, w, left=left, color=colors, edgecolor='none', height=bar_height)

                if show_pct_labels:
                    for rect, val, start, bar_color in zip(bars.patches, w, left, colors):
                        if val < min_pct_label:  # skip small values
                            continue
                        x_text = start + val / 2.0
                        text_color = get_text_color(bar_color)
                        ax.text(x_text, rect.get_y() + rect.get_height() / 2.0,
                                f"{val:.0f}%", va='center', ha='center', fontsize=annot_fontsize, color=text_color)
                left += w

            ax.set_yticks(y_pos)
            ax.set_yticklabels([y.title() for y in split_order])
            ax.tick_params(axis='y', labelsize=y_tick_fontsize)
            if not show_y_tick_labels:
                ax.set_yticklabels([])
                ax.tick_params(axis='y', length=0)
            if not shared_title:
                ax.set_title(f"{str(item_val)}", fontsize=title_fontsize)
            ax.grid(axis='x', linestyle=':', alpha=0.5)

    # Hide unused axes (if odd number of items in 2-col layout)
    for ax in axes_flat[n_items:]:
        ax.set_visible(False)

    # Shared x formatting
    for ax in axes_flat[:n_items]:
        ax.set_xticks(np.arange(0, 101, 20))
        ax.set_xticklabels([f"{int(x)}%" for x in np.arange(0, 101, 20)], fontsize=annot_fontsize)
        ax.set_xlim(0, 100)
        if show_x_per_subplot:
            ax.tick_params(axis='x', labelbottom=True)

    # Add x-axis label to bottom subplot(s) only
    if x_label:
        if x_label_fontsize is None:
            x_label_fontsize = title_fontsize
        if ncols == 2:
            # Label bottom of each column
            axes[nrows - 1, 0].set_xlabel(x_label, fontsize=x_label_fontsize)
            axes[nrows - 1, 1].set_xlabel(x_label, fontsize=x_label_fontsize)
        else:
            axes_flat[n_items - 1].set_xlabel(x_label, fontsize=x_label_fontsize)

    # legend
    # group legend
    def _rep_color_for_split(split_val):
        # choose the first available rating color for this split, else fall back
        if isinstance(color_maps.get(split_val, None), dict):
            for r in ratings_order:
                if r in color_maps[split_val]:
                    return color_maps[split_val][r]
        return "#888888"

    if show_group_legend:
        group_handles = [
            Line2D([0], [0],
                   marker='o', linestyle='None',
                   markerfacecolor=_rep_color_for_split(s),
                   markeredgecolor='white', markeredgewidth=0.8,
                   markersize=12, label=str(s).title())
            for s in split_order
        ]
        fig.legend(
            handles=group_handles,
            labels=[h.get_label() for h in group_handles],
            loc="upper left",
            bbox_to_anchor=(0.0, 1.0),
            frameon=False,
            ncol=1,
            borderaxespad=0.0,
            labelspacing=0.6,
            handletextpad=0.6
        )

    # item legend
    if legend is not None:
        base_group = next((g for g in split_order if g in color_maps and isinstance(color_maps[g], dict)), None)
        if base_group is not None:
            base_map = color_maps[base_group]
            order_for_legend = ratings_order if not legend_reverse else list(ratings_order)[::-1]

            handles = []
            for r in order_for_legend:
                if r not in base_map:
                    continue
                handles.append(
                    Line2D([0], [0],
                           marker='o', linestyle='None',
                           markerfacecolor=_to_gray(base_map[r]),
                           markeredgecolor='white', markeredgewidth=0.8,
                           markersize=legend_fontsize, label=f"{r}")
                )

            if handles:
                if legend == "inside":
                    axes[-1].legend(
                        handles=handles,
                        labels=[h.get_label() for h in handles],
                        loc='upper right',
                        frameon=False,
                        title="",
                        prop={'size': legend_fontsize}
                    )
                else:
                    # map the requested placement to a fig.legend location
                    legend_top_y = 0.92 if shared_title else 1.01  # lower legend when there's a shared title
                    legend_pos = {
                        "top": dict(loc="lower center", bbox_to_anchor=(0.5, legend_top_y), ncol=len(order_for_legend)),
                        "bottom": dict(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=len(order_for_legend)),
                        "right": dict(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1),
                        "left": dict(loc="center right", bbox_to_anchor=(-0.02, 0.5), ncol=1),
                    }.get(legend,
                          dict(loc="lower center", bbox_to_anchor=(0.5, legend_top_y), ncol=len(order_for_legend)))

                    fig.legend(
                        handles=handles,
                        labels=[h.get_label() for h in handles],
                        frameon=False,
                        title="",
                        prop={'size': legend_fontsize},
                        **legend_pos
                    )

    if shared_title:
        fig.suptitle(shared_title, fontsize=title_fontsize + 2, y=shared_title_y)

    # save plot
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(size_inches_x, size_inches_y)
    plt.savefig(os.path.join(save_path, f"{save_name}.{fmt}"), format=fmt, dpi=1000, bbox_inches="tight",
                pad_inches=0.01)
    del figure
    plt.clf()
    plt.close()
    return


def plot_c_ms(sub_df, save_path):
    ms_cols = list(survey_mapping.other_creatures_ms.values())
    c_cols = list(survey_mapping.other_creatures_cons.values())

    relevant_cols = {"Consciousness": c_cols, "Moral Status": ms_cols}
    for topic in relevant_cols:
        relevant_df = sub_df.loc[:, ["response_id", "sample"] + relevant_cols[topic]]
        relevant_df = pd.melt(relevant_df, id_vars=["sample", "response_id"], var_name="Item", value_name="Rating")
        relevant_df["Item"] = relevant_df["Item"].str.replace(r"^.*_", "", regex=True).replace(
            survey_mapping.other_creatures_general_names)
        relevant_df_grp = relevant_df.groupby(["sample", "Item"])["Rating"].agg(mean="mean", std="std").reset_index()
        relevant_df_grp.to_csv(os.path.join(save_path, f"group_hist_{topic}.csv"), index=False)
        # item_order=list(survey_mapping.other_creatures_general_names.values())
        item_order = (relevant_df_grp.groupby("Item")["mean"].mean().sort_values(ascending=False).index.tolist())

        # plot - per group (experiment)
        plot_hist_separate_groups(df=relevant_df, save_path=save_path, save_name=f"group_hist_{topic}",
                                  y_order=item_order, y_col="Item", facet="split",
                                  split_col="sample", split_order=["exploratory", "pre-registered", "follow-up"],
                                  rating_col="Rating", ratings_order=[1, 2, 3, 4],
                                  bar_height=0.93, bar_pitch=1.05,  # fatter bars
                                  color_maps=SAMPLE_COLORS_RATINGS,
                                  show_pct_labels=True,
                                  subplot_hspace=0.08,
                                  y_tick_fontsize=24,
                                  annot_fontsize=20,
                                  title_fontsize=25,
                                  fmt="svg", size_inches_x=20, size_inches_y=28,  # slightly shorter since less hspace
                                  legend="top", legend_fontsize=22, x_label="Proportion (%)", min_pct_label=4,  # hide anything under 5%
                                  show_x_per_subplot=True)

        # plot - per item
        plot_hist_separate_groups(df=relevant_df, save_path=save_path, save_name=f"item_hist_{topic}",
                                  y_order=item_order, y_col="Item", facet="item",
                                  split_col="sample", split_order=["follow-up", "pre-registered", "exploratory"],
                                  rating_col="Rating", ratings_order=[1, 2, 3, 4],
                                  bar_height=0.9, bar_pitch=1.2,
                                  color_maps=SAMPLE_COLORS_RATINGS,
                                  show_pct_labels=True, subplot_hspace=0.35, subplot_wspace=0.28,
                                  annot_fontsize=20,
                                  title_fontsize=22,
                                  fmt="svg", size_inches_x=20, size_inches_y=30,
                                  legend="top", legend_fontsize=22,
                                  show_y_tick_labels=True, show_group_legend=False, x_label="Proportion (%)",
                                  min_pct_label=4)  # hide anything under 5%
    return


def get_text_color(bg_hex):
    """
    Control for the text color (on bars etc) based on the luminance of what it appears on
    :param bg_hex: hex of the color of the patch
    :return: white if the color is dark, black if light
    """
    bg_hex = bg_hex.lstrip('#')  # Remove '#' and parse RGB
    r, g, b = int(bg_hex[0:2], 16), int(bg_hex[2:4], 16), int(bg_hex[4:6], 16)
    # Calculate relative luminance (perceived brightness)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "white" if luminance < 0.6 else "black"


def plot_binary_stacked_by_sample(df, questions, sample_col, sample_order, qa_order_map, sample_color_map,
                                  save_path, save_name, figsize=(10, 18), bar_height=0.65, bar_pitch=1.25,
                                  show_pct_labels=True, annot_fontsize=22, legend_labels=None,
                                  show_group_legend=False, show_sample_labels=False, x_tick_step=20,
                                  color_by_majority=True, majority_tie_policy="right",
                                  fmt="svg", dpi=1000, y_fontsize=22, x_fontsize=22,
                                  shared_title="", title_fontsize=24, title_y=1.08,
                                  x_label_perq=False, x_label_perq_names=None, x_label_perq_fontsize=20,
                                  x_label="", subplot_hspace=0.16,  # make sub-figures closer to one another
                                  left=0.14, right=0.90, bottom=0.06, top=0.94, alpha_major=1.0, alpha_minor=0.7,
                                  answer_color_map=None, show_side_answer_ticks=True, x_label_perq_pad=10,
                                  show_x_per_subplot=False, legend_mode="samples"  # or "answers"
                                  ):

    # figure scaffolding
    nrows = len(questions)

    def _count_samples_for_question(q):
        mapping = qa_order_map[q]
        mapped = df[q].map(mapping)
        temp = pd.DataFrame({"bin": mapped, "sample": df[sample_col]})
        count = 0
        for s in sample_order:
            sub = temp[temp["sample"] == s]
            if sub["bin"].notna().sum() > 0:
                count += 1
        return max(count, 1)  # at least 1 to avoid zero height

    height_ratios = [_count_samples_for_question(q) for q in questions]

    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=figsize,
                             sharex=(not show_x_per_subplot),
                             gridspec_kw=dict(hspace=subplot_hspace, left=left, right=right,
                                              bottom=bottom, top=top,
                                              height_ratios=height_ratios))
    if nrows == 1:
        axes = np.array([axes])

    def _compute_props_for_question(series, mapping):
        out = {}
        # map answers -> 0/1; unmapped become NaN and dropped from the denominator
        mapped = series.map(mapping)
        temp = pd.DataFrame({"bin": mapped, "sample": df.loc[series.index, sample_col]})  # combine with sample column aligned by index
        for s in sample_order:
            sub = temp[temp["sample"] == s]
            # count valid 0/1 only (ignore NaN)
            counts = sub["bin"].value_counts(dropna=True)
            total = counts.sum()
            if total == 0:
                continue  # SKIP samples with no answers to the question
            else:
                p0 = float(counts.get(0, 0)) * 100.0 / total
                p1 = float(counts.get(1, 0)) * 100.0 / total
                # ensure 0..100 bounds and rounding consistency
                out[s] = {0: max(0.0, min(100.0, p0)), 1: max(0.0, min(100.0, p1))}
        return out

    def _label_for(ans_label):
        if legend_labels is None:
            return str(ans_label)
        return str(legend_labels.get(ans_label, ans_label))

    def _colors_for_sample(s):
        # returns (left_color, right_color)
        entry = sample_color_map.get(s, {0: "#bdbdbd", 1: "#737373"})
        if isinstance(entry, dict):
            return entry.get(0, "#bdbdbd"), entry.get(1, "#737373")
        return tuple(entry)

    # plot
    for ax, q in zip(axes, questions):
        mapping = qa_order_map[q]  # {answer_label -> 0/1}
        # find which labels correspond to 0 and 1 to title the side ticks
        left_label_raw = next((lbl for lbl, z in mapping.items() if z == 0), None)
        right_label_raw = next((lbl for lbl, z in mapping.items() if z == 1), None)
        left_label = _label_for(left_label_raw)
        right_label = _label_for(right_label_raw)
        props = _compute_props_for_question(df[q], mapping)  # {sample -> {0: pct, 1: pct}}

        # only plot samples that actually have data for this question
        present_samples = [s for s in sample_order if s in props]
        if len(present_samples) == 0:
            # nothing to draw for this question; keep axis clean but intact
            ax.set_yticks([])
            if x_label_perq:
                per_ax_title = x_label_perq_names[q]
                ax.set_title(per_ax_title, fontsize=x_label_perq_fontsize, pad=x_label_perq_pad)
            ax.grid(axis="x", linestyle=":", alpha=0.45)
            continue

        y_pos = np.arange(len(present_samples)) * bar_pitch
        ax.set_ylim(y_pos[0] - bar_height / 2, y_pos[-1] + bar_height / 2)

        # Draw bars per sample - only for present samples
        for yi, s in enumerate(present_samples):
            p0 = props[s][0]
            p1 = props[s][1]

            # base colors for this sample (safe even if s missing in sample_color_map)
            base_left, base_right = _colors_for_sample(s)

            if color_by_majority:
                if np.isclose(p0, p1):
                    majority_is_right = (majority_tie_policy == "right")
                else:
                    majority_is_right = (p1 > p0)

                if majority_is_right:
                    color_left = base_left  # left segment when right is majority gets the '0' color
                    color_right = base_right  # right segment gets the '1' color
                    alpha_left, alpha_right = alpha_minor, alpha_major
                else:
                    color_left = base_right  # flip when left is majority
                    color_right = base_left
                    alpha_left, alpha_right = alpha_major, alpha_minor
            else:
                color_left, color_right = base_left, base_right
                alpha_left = alpha_right = alpha_major

            if answer_color_map is not None:
                color_left = answer_color_map.get(left_label_raw, color_left)
                color_right = answer_color_map.get(right_label_raw, color_right)

            # Left segment (0)
            ax.barh(y_pos[yi], p0, left=0.0, height=bar_height, color=color_left, edgecolor="none", alpha=alpha_left)
            # Right segment (1)
            ax.barh(y_pos[yi], p1, left=p0, height=bar_height, color=color_right, edgecolor="none", alpha=alpha_right)

            if show_pct_labels:
                left_text_color = get_text_color(color_left)
                ax.text(p0 / 2.0, y_pos[yi], f"{p0:.0f}%", va="center", ha="center",
                        color=left_text_color, fontsize=annot_fontsize)
                right_text_color = get_text_color(color_right)
                ax.text(p0 + p1 / 2.0, y_pos[yi], f"{p1:.0f}%", va="center", ha="center",
                        color=right_text_color, fontsize=annot_fontsize)

        # labels (left/right)
        mid_y = float(np.mean(y_pos)) if len(y_pos) > 1 else y_pos[0]
        ax.set_yticks([])

        # if we want to show for each subplot (=question) the answers on the ticks themselves (left and right side)
        if show_side_answer_ticks:
            from textwrap import wrap
            # RIGHT label (for '1')
            a_right = ax.twinx()
            a_right.set_ylim(ax.get_ylim())
            a_right.set_yticks([mid_y])
            right_label_wrapped = "\n".join(wrap(str(right_label), width=9, break_long_words=False, break_on_hyphens=False)[:2])
            a_right.set_yticklabels([right_label_wrapped], fontsize=x_fontsize, ha="left")
            a_right.tick_params(axis="y", labelright=True, labelleft=False, length=0, pad=5)
            for spine in ["left", "right", "top", "bottom"]:
                a_right.spines[spine].set_visible(False)

            # LEFT label (for '0') as a secondary y-axis with identity transform
            a_left = ax.secondary_yaxis('left', functions=(lambda y: y, lambda y: y))
            a_left.set_ylim(ax.get_ylim())
            a_left.set_yticks([mid_y])
            left_label_wrapped = "\n".join(wrap(str(left_label), width=9, break_long_words=False, break_on_hyphens=False)[:2])
            a_left.set_yticklabels([left_label_wrapped], fontsize=x_fontsize, ha="right")
            a_left.tick_params(axis="y", labelleft=True, labelright=False, length=0, pad=5)
            for spine in ["left", "right", "top", "bottom"]:
                a_left.spines[spine].set_visible(False)

        if show_sample_labels:
            ax.set_yticks(y_pos)
            ax.set_yticklabels([str(s).title() for s in present_samples], fontsize=y_fontsize)
        else:
            ax.set_yticks([])

        ax.grid(axis="x", linestyle=":", alpha=0.45)

        if x_label_perq:
            per_ax_title = x_label_perq_names[q]
            ax.set_title(per_ax_title, fontsize=x_label_perq_fontsize, pad=x_label_perq_pad)

    # shared x-axis formatting
    for ax in axes:
        ax.set_xlim(0, 100)
        ticks = np.arange(0, 101, x_tick_step)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{int(t)}" for t in ticks], fontsize=x_fontsize)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        axes_tight_top = max(
            ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted()).y1
            for ax in axes
        )
        if getattr(fig, "_suptitle", None) is not None:
            sup_bbox_fig = fig._suptitle.get_window_extent(renderer).transformed(fig.transFigure.inverted())
            sup_bottom = sup_bbox_fig.y0
        else:
            sup_bottom = 1.0

    if show_group_legend:
        handles = []
        for s in sample_order:
            _, right_col = _colors_for_sample(s)
            handles.append(
                Line2D([0], [0], marker="o", linestyle="None",
                       markerfacecolor=right_col, markeredgecolor="white",
                       markeredgewidth=0.8, markersize=x_fontsize - 2, label=str(s).title())
            )
        fig.legend(handles=handles,
                   labels=[h.get_label().title() for h in handles],
                   loc="lower center", bbox_to_anchor=(0.5, top + 0.03),
                   ncol=len(sample_order), frameon=False,
                   fontsize=x_fontsize - 2)

    if legend_mode == "answers":
        first_q = questions[0]
        mapping0 = qa_order_map[first_q]
        left_ans = next((lbl for lbl, z in mapping0.items() if z == 0), None)
        right_ans = next((lbl for lbl, z in mapping0.items() if z == 1), None)
        left_lab, right_lab = _label_for(left_ans), _label_for(right_ans)

        if answer_color_map is not None:
            c0 = answer_color_map.get(left_ans, "#999999")
            c1 = answer_color_map.get(right_ans, "#666666")
        else:
            any_s = next((s for s in sample_order if s in sample_color_map), None)
            if any_s is not None:
                c0, c1 = _colors_for_sample(any_s)
            else:
                c0, c1 = ("#bdbdbd", "#737373")

        ans_handles = [
            Line2D([0], [0], marker="o", linestyle="None",
                   markerfacecolor=c0, markeredgecolor="white",
                   markeredgewidth=0.8, markersize=x_fontsize - 2, label=left_lab),
            Line2D([0], [0], marker="o", linestyle="None",
                   markerfacecolor=c1, markeredgecolor="white",
                   markeredgewidth=0.8, markersize=x_fontsize - 2, label=right_lab)
        ]
        fig.legend(handles=ans_handles, labels=[left_lab, right_lab],
                   loc="lower center", bbox_to_anchor=(0.5, top + 0.03),  # same as show_group_legend
                   ncol=2, frameon=False,
                   fontsize=x_fontsize - 2)

    fig.suptitle(shared_title.title(), fontsize=title_fontsize, y=title_y)
    fig.supxlabel(x_label, fontsize=title_fontsize)

    # save
    os.makedirs(save_path, exist_ok=True)
    fig.set_size_inches(figsize[0], figsize[1])
    out_path = os.path.join(save_path, f"{save_name}.{fmt}")
    plt.savefig(out_path, format=fmt, dpi=dpi, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    return


def plot_eid(sub_df, save_path):
    relevant_df = sub_df[sub_df["sample"] != "follow-up"]
    eid_questions = list(survey_mapping.EARTH_DANGER_QA_MAP.keys())
    relevant_df = relevant_df.loc[:, ["response_id", "sample"] + eid_questions]

    plot_binary_stacked_by_sample(df=relevant_df, questions=eid_questions,
                                  sample_col="sample", sample_order=["pre-registered", "exploratory"],  # exploratory at the TOP
                                  qa_order_map=survey_mapping.EARTH_DANGER_QA_MAP,
                                  sample_color_map=SAMPLE_COLORS_SELECTED_NO, color_by_majority=True,
                                  save_path=save_path, save_name="bars_eid", figsize=(22, 12),
                                  bar_height=1.2, bar_pitch=1.3,
                                  show_pct_labels=True,
                                  legend_labels=survey_mapping.EARTH_DANGER_ANS_MAP,
                                  show_group_legend=True,
                                  show_sample_labels=False,  # set True if you want y-axis sample names too
                                  subplot_hspace=0.32,
                                  annot_fontsize=26,
                                  x_fontsize=26,
                                  y_fontsize=26,
                                  title_fontsize=28, title_y=1.08,
                                  x_tick_step=20, fmt="svg", dpi=1000, top=0.90, bottom=0.08,
                                  shared_title="Who would you save?", x_label="Proportion (%)")
    return


def plot_ics(sub_df, save_path):
    relevant_df = sub_df[sub_df["sample"] != "follow-up"]
    ics_questions = list(survey_mapping.ICS_Q_NAME_MAP.keys())
    relevant_df = relevant_df.loc[:, ["response_id", "sample"] + ics_questions]

    fig_title = "Do you think a creature/system can "
    ics_question_titles = {x: x.removeprefix(fig_title).capitalize() for x in ics_questions}

    plot_binary_stacked_by_sample(df=relevant_df, questions=ics_questions,
                                  sample_col="sample", sample_order=["pre-registered", "exploratory"],  # exploratory at the TOP
                                  qa_order_map={q: {"Yes": 0, "No": 1} for q in ics_questions},
                                  sample_color_map=SAMPLE_COLORS_SELECTED_NO,
                                  answer_color_map=YESNO_COLORS,
                                  color_by_majority=False,  # by side (yes/no always the same color)
                                  show_side_answer_ticks=False,  # COLOR BY THE ANSWER (otherwise=True)
                                  show_sample_labels=True,  # set True if you want y-axis sample names too
                                  legend_mode="answers",  # change to "samples" to color by the samples
                                  show_group_legend=False,  # change to "True" to do sample colors
                                  save_path=save_path, save_name="bars_ics", figsize=(20, 12),
                                  bar_height=0.90, bar_pitch=1.00,
                                  show_pct_labels=True,
                                  legend_labels=None,
                                  x_label_perq=True,
                                  x_label_perq_names=ics_question_titles,
                                  x_label_perq_fontsize=26,
                                  x_label_perq_pad=12,
                                  subplot_hspace=0.35,
                                  y_fontsize=24, x_fontsize=26, annot_fontsize=26,
                                  alpha_major=1.0, alpha_minor=1.0,
                                  title_fontsize=28,
                                  fmt="svg", dpi=1000,
                                  left=0.14, right=0.90, bottom=0.08, top=0.90,
                                  shared_title=fig_title, x_label="Proportion (%)")


    # bar per sample, COLOR PER SAMPLE, ANS ON AXIS
    #plot_binary_stacked_by_sample(df=relevant_df, questions=ics_questions,
    #                              sample_col="sample", sample_order=["pre-registered", "exploratory"],
    #                              qa_order_map={q: {"Yes": 0, "No": 1} for q in ics_questions},
    #                              # Yes on the left always
    #                              sample_color_map=SAMPLE_COLORS_SELECTED_NO,
    #                              color_by_majority=False,  # by side (yes/no always the same color)
    #                              save_path=save_path, save_name="bars_ics", figsize=(20, 12),
    #                              bar_height=1.2, bar_pitch=1.3,
    #                              show_pct_labels=True,
    #                              legend_labels=None,
    #                              show_group_legend=True,
    #                              x_label_perq=True,  # show the question names here
    #                              x_label_perq_names=ics_question_titles,
    #                              subplot_hspace=0.35,
    #                              y_fontsize=25, x_fontsize=22, annot_fontsize=22, alpha_major=1.0, alpha_minor=1.0,
    #                              show_sample_labels=False,  # set True if you want y-axis sample names too
    #                              x_tick_step=20, fmt="svg", dpi=1000,
    #                              left=0.14, right=0.90, bottom=0.06, top=0.90,  # top: smaller=more space
    #                              shared_title=fig_title, x_label="Proportion (%)")
    return


def plot_kpt(sub_df, save_path):
    relevant_df = sub_df[sub_df["sample"] != "follow-up"]
    kpt_questions = list(survey_mapping.important_test_kill_tokens.keys())
    relevant_df = relevant_df.loc[:, ["response_id", "sample"] + kpt_questions]
    ans_map = {survey_mapping.ANS_KILL: "Yes", survey_mapping.ANS_NOKILL: "No"}
    relevant_df[kpt_questions] = relevant_df[kpt_questions].replace(ans_map)

    prefix = "A creature/system that "
    #kpt_question_titles = {x: x.removeprefix(prefix).capitalize() for x in kpt_questions}

    fig_title = "Would you kill to pass the test?"
    plot_binary_stacked_by_sample(df=relevant_df, questions=kpt_questions,
                                  sample_col="sample", sample_order=["pre-registered", "exploratory"],  # exploratory at the TOP
                                  qa_order_map={q: {"Yes": 0, "No": 1} for q in kpt_questions},
                                  # Yes on the left always
                                  sample_color_map=SAMPLE_COLORS_SELECTED_NO,
                                  answer_color_map=YESNO_COLORS,
                                  # to color by the SAMPLE: show_side..True, show_sample..=False, legend_mode="samples", show_group..=True
                                  color_by_majority=False,  # by side (yes/no always the same color)
                                  show_side_answer_ticks=False,  # COLOR BY THE ANSWER (otherwise=True)
                                  show_sample_labels=True,  # set True if you want y-axis sample names too
                                  legend_mode="answers",  # change to "samples" to color by the samples
                                  show_group_legend=False,  # change to "True" to do sample colors
                                  save_path=save_path, save_name="bars_kpt", figsize=(20, 12),
                                  bar_height=1.2, bar_pitch=1.3,
                                  show_pct_labels=True,
                                  legend_labels=None,
                                  x_label_perq=True,  # show the question names here
                                  x_label_perq_names=survey_mapping.important_test_kill_tokens,
                                  x_label_perq_fontsize=24,
                                  subplot_hspace=0.6,  # spacing between subplots
                                  y_fontsize=24, x_fontsize=26, annot_fontsize=26, alpha_major=1.0, alpha_minor=1.0,
                                  title_fontsize=25, x_tick_step=20, fmt="svg", dpi=1000,
                                  left=0.14, right=0.90, bottom=0.07, top=0.89,  # top: smaller=more space
                                  shared_title=fig_title, x_label="Proportion (%)")
    return


def plot_zombie(sub_df, save_path):
    relevant_df = sub_df[sub_df["sample"] != "follow-up"]
    plot_pie(df=relevant_df, save_path=save_path, save_name="pie_zombie", fmt="svg",
             stat_col=survey_mapping.Q_ZOMBIE,
             split_col="sample", split_order=["exploratory", "pre-registered"],  # first is LEFT
             stat_order=["Yes", "No"],
             color_palette=YESNO_COLORS, shared_title=survey_mapping.Q_ZOMBIE.title(),
             size_inches_x=12, size_inches_y=6, title_pad=12,
             annot_fontsize=20, title_fontsize=22, legend_fontsize=20)
    return


def plot_moral_consideration_features(sub_df, save_path):
    x_col = survey_mapping.Q_FEATURES_IMPORTANT
    y_col = "sample"
    df_exp = sub_df.loc[:, ["response_id", x_col, y_col]]
    df_exp = df_exp[df_exp[x_col].notnull()]  # all people answered this, we shouldn't lose anyone here

    valid_answers = survey_mapping.ALL_FEATURES

    def parse_multiselect(val):
        """Extract valid answers from a comma-separated string, handling commas within answers."""
        if isinstance(val, (list, tuple, set)):
            return list(val)
        val_str = str(val)
        found = [ans for ans in valid_answers if ans in val_str]
        return found if found else []

    df_exp = df_exp.assign(**{x_col: lambda d: d[x_col].apply(parse_multiselect)}).explode(x_col).loc[lambda d: d[x_col].notna() & d[x_col].ne("")]

    # numerator
    ppl_per_topic = (df_exp[["response_id", y_col, x_col]].drop_duplicates().
                     groupby([y_col, x_col], dropna=False).size().reset_index(name="n_people"))
    # denominator
    denom = df_exp.groupby(y_col)["response_id"].nunique().rename("n_people_total").reset_index()

    # merge & compute percent-of-people
    counts = ppl_per_topic.merge(denom, on=y_col, how="left")
    counts["percent"] = 100 * counts["n_people"] / counts["n_people_total"]

    plot_hist_multiselect(df=counts, save_path=save_path, save_name=f"hist_ms_features", fmt="svg",
                          x_col=x_col, y_col=y_col, y_order=["exploratory", "pre-registered", "follow-up"],
                          title=survey_mapping.Q_FEATURES_IMPORTANT.title(), y_title="Percent (%)", x_title=f"",
                          n_col="n_people", denom_col="n_people_total",
                          metric="percent",
                          size_inches_x=16, size_inches_y=8, alpha_val=1.0, color_palette=SAMPLE_COLORS,
                          x_ticks=None, y_ticks=np.arange(0, 101, 10), x_tick_rotation=30, bar_shrink=0.8,
                          x_text_width=22)

    counts.to_csv(os.path.join(save_path, "hist_ms_features.csv"), index=False)
    return


def plot_moral_prios(sub_df, save_path):
    prios_qs = list(survey_mapping.PRIOS_Q_NAME_MAP.keys())
    relevant_df = sub_df.loc[:, ["response_id", "sample"] + prios_qs]
    prios_labels = {q: q.capitalize() for q in prios_qs}
    plot_binary_stacked_by_sample(df=relevant_df, questions=prios_qs,
                                  sample_col="sample", sample_order=["follow-up", "pre-registered", "exploratory"],
                                  qa_order_map={q: {"Yes": 0, "No": 1} for q in survey_mapping.PRIOS_Q_NAME_MAP},
                                  # Yes on the left always
                                  sample_color_map=SAMPLE_COLORS_SELECTED_NO,
                                  answer_color_map=YESNO_COLORS,
                                  # to color by the SAMPLE: show_side..True, show_sample..=False, legend_mode="samples", show_group..=True
                                  color_by_majority=False,  # by side (yes/no always the same color)
                                  show_side_answer_ticks=False,  # COLOR BY THE ANSWER (otherwise=True)
                                  show_sample_labels=True,  # set True if you want y-axis sample names too
                                  legend_mode="answers",  # change to "samples" to color by the samples
                                  show_group_legend=False,  # change to "True" to do sample colors
                                  save_path=save_path, save_name="bars_prios", figsize=(20, 12),
                                  bar_height=1.2, bar_pitch=1.3,
                                  show_pct_labels=True,
                                  legend_labels=None,
                                  x_label_perq=True,  # show the question names here
                                  x_label_perq_names=prios_labels,
                                  x_label_perq_fontsize=24,
                                  subplot_hspace=0.6,  # spacing between subplots
                                  y_fontsize=18, x_fontsize=22, annot_fontsize=22, alpha_major=1.0, alpha_minor=1.0,
                                  title_fontsize=25, x_tick_step=20, fmt="svg", dpi=1000,
                                  left=0.14, right=0.90, bottom=0.08, top=0.89,  # top: smaller=more space
                                  shared_title="Moral Considerations and Consciousness",
                                  x_label="Proportion (%)", show_x_per_subplot=True)
    return


def plot_graded_c(sub_df, save_path):
    graded_qs = list(survey_mapping.Q_GRADED_NAMES.keys())
    relevant_df = sub_df.loc[:, ["response_id", "sample"] + graded_qs]
    long_df = relevant_df.melt(id_vars=["response_id", "sample"], value_vars=graded_qs,
                               var_name="question", value_name="Rating").reset_index(drop=True)

    plot_hist_separate_groups(df=long_df, save_path=save_path, save_name=f"hist_gradedC",
                              y_order=graded_qs, y_col="question", facet="item",
                              split_col="sample", split_order=["pre-registered", "exploratory"],
                              rating_col="Rating", ratings_order=[1, 2, 3, 4],
                              bar_height=0.5, bar_pitch=0.55,
                              color_maps=SAMPLE_COLORS_RATINGS,
                              show_pct_labels=True,
                              subplot_hspace=0.80,
                              subplot_wspace=0.25,
                              fmt="svg",
                              size_inches_x=20, size_inches_y=8,
                              legend="top",
                              legend_fontsize=24,
                              annot_fontsize=25,
                              title_fontsize=29,
                              y_tick_fontsize=25,
                              show_y_tick_labels=True,
                              show_group_legend=False,
                              legend_reverse=False,
                              min_pct_label=4,
                              x_label="Proportion (%)", show_x_per_subplot=True,
                              shared_title="Graded Consciousness")

    return


def plot_intelligence(sub_df, save_path):
    relevant_df = sub_df[sub_df["sample"] != "follow-up"]
    relevant_df = relevant_df.loc[:, ["response_id", "sample", survey_mapping.Q_INTELLIGENCE, survey_mapping.Q_INTELLIGENCE_HOW]]

    # use just the follow-up column: if nan, people said "no" (unrelated); and otherwise it's the "how" for the "yes"
    relevant_df[survey_mapping.Q_INTELLIGENCE_HOW] = np.where(relevant_df[survey_mapping.Q_INTELLIGENCE] == "No",
                                                              relevant_df[survey_mapping.Q_INTELLIGENCE_HOW].fillna("No"),
                                                              relevant_df[survey_mapping.Q_INTELLIGENCE_HOW])

    plot_pie(df=relevant_df, save_path=save_path, save_name="pie_intelligence", fmt="svg",
             stat_col=survey_mapping.Q_INTELLIGENCE_HOW,
             split_col="sample", split_order=["pre-registered", "exploratory"],
             stat_order=C_I_HOW_ORDER, legend_fontsize=20,
             color_palette=C_I_HOW_COLOR_MAP,
             size_inches_x=12, size_inches_y=8, text_width=28, shared_title=survey_mapping.Q_INTELLIGENCE.title())

    return


def manage_plotting(sub_df, save_path):
    """
    Manage all plots for the paper supplementary including all the data
    :param sub_df: df unified with all samples in it
    :param save_path: save_path
    """

    warnings.filterwarnings("ignore", category=FutureWarning)

    """
    Plot moral status / consciousness ratings per entity / group across three surveys
    """
    plot_c_ms(sub_df, save_path)

    """
    Correlation plot between consciousness and moral status - overlaid across all three samples
    """
    plot_c_v_ms(sub_df, save_path)

    """
    Earth in danger (EiD): just the two samples of the main
    """
    plot_eid(sub_df, save_path)

    """
    Intentions, Consciousness, Sensations (valence) seperability  ***  >>> still need to do the follow-up
    """
    plot_ics(sub_df, save_path)

    """
    Kill to pass test (KPT): just the two samples of the main     ***  >>> still need to do the follow-up
    """
    plot_kpt(sub_df, save_path)

    """
    Zombie pill: just the two samples of the main
    """
    plot_zombie(sub_df, save_path)

    """
    Moral consideration features: all three samples
    """
    plot_moral_consideration_features(sub_df, save_path)

    """
    Moral priorities block
    """
    plot_moral_prios(sub_df, save_path)

    """
    Graded consciousness block: just the two samples of the main 
    """
    plot_graded_c(sub_df, save_path)

    """
    Consciousness and intelligence
    """
    plot_intelligence(sub_df, save_path)

    """
    Demographics
    """
    plot_demographics(df=sub_df, save_path=save_path)

    """
    Experience 
    """
    plot_experience(sub_df, save_path)
    plot_experience_followup(sub_df, save_path)

    return


def c_v_ms_colored_by_offDiag(preregistered_path, save_path):
    """
    Figure 1: c-v-ms correlation scatter only (no diff panel)
    """
    long_data = pd.read_csv(os.path.join(preregistered_path, "c_v_ms", "c_v_ms_long.csv"))
    item_diff = pd.read_csv(os.path.join(preregistered_path, "c_v_ms", "item_off_diagonal_differences.csv"))

    result_path = os.path.join(save_path, "fig 1")

    # Prepare data
    df_pivot = long_data.pivot_table(index="Item", columns="Topic", values="Rating", aggfunc="mean").reset_index()
    df_pivot["Item"] = df_pivot["Item"].replace(survey_mapping.other_creatures_general_names)

    # Compute colors for item_diff
    item_diff["color"] = np.select(
        [
            (item_diff["off_diagonal"]) & (item_diff["diff"] > 0),
            (item_diff["off_diagonal"]) & (item_diff["diff"] < 0)
        ],
        [1, -1],
        default=0
    )
    item_diff["Item"] = item_diff["Item"].replace(survey_mapping.other_creatures_general_names)

    # Merge color into scatter df
    df_pivot = df_pivot.merge(item_diff[["Item", "color"]], on="Item", how="left")
    df_pivot["color"] = df_pivot["color"].fillna(0).astype(int)

    # Colors
    colors_dict = {
        0: "#e5e5e5",
        1: YESNO_COLORS[survey_mapping.ANS_YES],
        -1: YESNO_COLORS[survey_mapping.ANS_NO]
    }

    # Icon paths
    short_name_to_icon = {
        survey_mapping.other_creatures_general_names[k]: v
        for k, v in survey_mapping.other_creatures_icon_paths.items()
    }

    label_names = {
        1: "Does\nNot Have",
        2: "Probably Doesn't Have",
        3: "Probably Has",
        4: "Has"
    }

    plot_scatter_icons(
        df=df_pivot,
        x_col="Consciousness", y_col="Moral Status", identity_col="Item",
        group_col="color", group_order=[1, 0, -1],
        group_colors=colors_dict,
        icon_paths=short_name_to_icon,
        x_range=np.arange(1, 5, 1), y_range=np.arange(1, 5, 1),
        x_tick_labels=label_names,
        y_tick_labels=label_names,
        y_tick_rotation=90,
        icon_size=70, icon_max_width=70,  # bigger icons
        alpha=0.95,
        border_width=0.5, border_color=(0, 0, 0),
        vertical_jitter=0, horizontal_jitter=0,
        size_inches_x=20, size_inches_y=16,  # same width, taller for square-ish plot
        fmt="svg",
        save_path=result_path, save_name="fig1_scatter_only",
        legend_fontsize=24,
        axis_label_fontsize=28,
        tick_fontsize=24,
        diag_line=True,
        show_annotations=False,
        show_group_legend=False,  # no color legend
        show_icon_legend=True,  # yes icon legend
        icon_legend_size=50,
        icon_legend_max_width=50,
        icon_legend_fontsize=22)

    return



def plot_scatter_with_differences(df_scatter, df_diff,
                                  x_col, y_col, identity_col,
                                  diff_value_col, diff_category_col,
                                  group_col, group_colors,
                                  icon_paths, save_path, save_name,
                                  x_range, y_range,
                                  # scatter params
                                  x_tick_labels=None, y_tick_labels=None, y_tick_rotation=0,
                                  icon_size=50, icon_max_width=None, alpha=0.85,
                                  vertical_jitter=0, horizontal_jitter=0,
                                  diag_line=True, show_annotations=False,
                                  border_width=0, border_color=(50, 50, 50),
                                  # diff params
                                  se=False, se_col=None, diff_alpha=1.0, diff_bar_height=0.7, diff_xlim=None,
                                  # layout params
                                  size_inches_x=20, size_inches_y=12,
                                  scatter_width_ratio=0.55,  # fraction of figure for scatter
                                  # font params
                                  axis_label_fontsize=22, tick_fontsize=18,
                                  diff_title="", diff_title_fontsize=20,
                                  diff_icon_size=20, diff_icon_max_width=20,
                                  # identity mapping
                                  identity_name_map=None,
                                  fmt="svg", diff_label_fontsize=14):
    """
    Combined scatter plot with icons (left) and off-diagonal differences with icons (right).
    """
    df_scatter = df_scatter.copy()
    df_diff = df_diff.copy()

    # Apply name mapping if provided
    if identity_name_map:
        df_scatter[identity_col] = df_scatter[identity_col].replace(identity_name_map)
        df_diff[identity_col] = df_diff[identity_col].replace(identity_name_map)

    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    # Create figure with two subplots
    fig = plt.figure(figsize=(size_inches_x, size_inches_y))

    # GridSpec for custom widths
    gs = fig.add_gridspec(1, 2, width_ratios=[scatter_width_ratio, 1 - scatter_width_ratio],
                          wspace=0.08, left=0.06, right=0.98, top=0.95, bottom=0.08)

    ax_scatter = fig.add_subplot(gs[0])
    ax_diff = fig.add_subplot(gs[1])

    # ==================== LEFT: SCATTER PLOT ====================
    x_min, x_max = min(x_range) - 0.1, max(x_range) + 0.25
    y_min, y_max = min(y_range) - 0.1, max(y_range) + 0.25
    ax_scatter.set_xlim(x_min, x_max)
    ax_scatter.set_ylim(y_min, y_max)

    # Diagonal line
    if diag_line:
        start, end = min(x_range), max(x_range)
        ax_scatter.plot([start, end], [start, end], color="#1F2041", linestyle="dashed",
                        linewidth=1.25, zorder=2)

    # Jitter
    counts = df_scatter.groupby([x_col, y_col]).size().reset_index(name="counts")
    jitter_mask = counts[counts["counts"] > 1].set_index([x_col, y_col]).index

    if vertical_jitter > 0 or horizontal_jitter > 0:
        for i in range(len(df_scatter)):
            if (df_scatter[x_col].iat[i], df_scatter[y_col].iat[i]) in jitter_mask:
                if vertical_jitter > 0:
                    df_scatter[y_col].iat[i] += np.random.uniform(-vertical_jitter, vertical_jitter)
                if horizontal_jitter > 0:
                    df_scatter[x_col].iat[i] += np.random.uniform(-horizontal_jitter, horizontal_jitter)

    # Load icons for scatter
    icon_scale = 8
    icon_cache = {}
    for identity, path in icon_paths.items():
        if os.path.exists(path):
            icon_cache[identity] = load_and_normalize_icon(path, icon_size * icon_scale,
                                                           icon_max_width * icon_scale if icon_max_width else None)

    # plot scatter icons
    for i in range(len(df_scatter)):
        x = df_scatter[x_col].iat[i]
        y = df_scatter[y_col].iat[i]
        identity = df_scatter[identity_col].iat[i]
        group = df_scatter[group_col].iat[i]

        if identity in icon_cache and group in group_colors:
            base_img = icon_cache[identity]
            tinted_img = tint_image(base_img, group_colors[group],
                                    border_width=int(border_width * icon_scale),
                                    border_color=border_color)
            tinted_img = tinted_img.copy()
            tinted_img[:, :, 3] = (tinted_img[:, :, 3] * alpha).astype(np.uint8)

            imagebox = OffsetImage(tinted_img, zoom=1.0 / icon_scale)
            imagebox.image.axes = ax_scatter
            ab = AnnotationBbox(imagebox, (x, y), frameon=False, zorder=3)
            ax_scatter.add_artist(ab)

    # scatter axis labels
    ax_scatter.set_xlabel(x_col, fontsize=axis_label_fontsize, labelpad=10)
    ax_scatter.set_ylabel(y_col, fontsize=axis_label_fontsize, labelpad=8)
    ax_scatter.set_xticks(x_range)
    ax_scatter.set_yticks(y_range)

    # customize (if provided)
    if x_tick_labels:
        labels = [x_tick_labels.get(t, str(t)) for t in x_range]
        ax_scatter.set_xticklabels(labels, fontsize=tick_fontsize)
    else:
        ax_scatter.tick_params(axis='x', labelsize=tick_fontsize)

    if y_tick_labels:
        labels = [y_tick_labels.get(t, str(t)) for t in y_range]
        ax_scatter.set_yticklabels(labels, fontsize=tick_fontsize, rotation=y_tick_rotation, va='center', ha='right')
    else:
        ax_scatter.tick_params(axis='y', labelsize=tick_fontsize)

    # annotations on scatter
    if show_annotations:
        eps = 0.05
        keep_mask = dedup_annotation_indices(df_scatter, x_col, y_col, identity_col, eps=eps)
        kept_indices = np.flatnonzero(keep_mask)
        for i in kept_indices:
            ax_scatter.text(df_scatter[x_col].iat[i], df_scatter[y_col].iat[i] + 0.15,
                            df_scatter[identity_col].iat[i], fontsize=14, ha="center")

    # ==================== RIGHT: DIFFERENCES WITH ICONS ====================
    df_diff_sorted = df_diff.sort_values(diff_value_col)

    colors = df_diff_sorted[diff_category_col].map(group_colors)
    y_positions = np.arange(len(df_diff_sorted))

    # Draw bars
    bars = ax_diff.barh(y_positions, df_diff_sorted[diff_value_col],
                        color=colors, alpha=diff_alpha, height=diff_bar_height)

    # Error bars
    if se and se_col:
        ax_diff.errorbar(df_diff_sorted[diff_value_col], y_positions,
                         xerr=1.96 * df_diff_sorted[se_col],
                         fmt='none', ecolor='black', elinewidth=1.2,
                         capsize=3, capthick=1.2)

    if diff_xlim:
        ax_diff.set_xlim(diff_xlim)

    # Vertical line at 0
    ax_diff.axvline(0, color="black", linestyle="--", linewidth=1)

    # Load icons for diff plot (smaller)
    diff_icon_scale = 8
    diff_icon_cache = {}
    for identity, path in icon_paths.items():
        if os.path.exists(path):
            diff_icon_cache[identity] = load_and_normalize_icon(
                path, diff_icon_size * diff_icon_scale,
                diff_icon_max_width * diff_icon_scale if diff_icon_max_width else None)

    # Add icons and text as y-tick labels
    ax_diff.set_yticks(y_positions)
    ax_diff.set_yticklabels([])  # Remove default text labels

    # Get axis limits for positioning
    x_lim = ax_diff.get_xlim()
    icon_x_position = x_lim[0] - (x_lim[1] - x_lim[0]) * 0.05  # Icons close to axis
    text_x_position = x_lim[0] - (x_lim[1] - x_lim[0]) * 0.17  # Text to left of icons

    for i, (idx, row) in enumerate(df_diff_sorted.iterrows()):
        identity = row[identity_col]
        category = row[diff_category_col]

        # Add icon
        if identity in diff_icon_cache:
            base_img = diff_icon_cache[identity]
            tinted_img = tint_image(base_img, group_colors.get(category, "#888888"),
                                    border_width=int(border_width * diff_icon_scale),
                                    border_color=border_color)

            imagebox = OffsetImage(tinted_img, zoom=1.0 / diff_icon_scale)
            imagebox.image.axes = ax_diff
            ab = AnnotationBbox(imagebox, (icon_x_position, i), frameon=False,
                                xycoords=('data', 'data'), box_alignment=(1.0, 0.5),
                                clip_on=False)
            ax_diff.add_artist(ab)

        # Add text label to the left of icon
        ax_diff.text(text_x_position, i, identity,
                     fontsize=diff_label_fontsize, ha='right', va='center',
                     clip_on=False)

    # Diff axis formatting
    ax_diff.set_xlabel("Difference", fontsize=axis_label_fontsize, labelpad=10)
    ax_diff.tick_params(axis='x', labelsize=tick_fontsize)
    ax_diff.tick_params(axis='y', length=0)  # Hide y-tick marks

    if diff_title:
        ax_diff.set_title(diff_title, fontsize=diff_title_fontsize)

    # adjust x-limits to make room for text + icons
    if diff_xlim:
        xlim_range = diff_xlim[1] - diff_xlim[0]
        ax_diff.set_xlim(diff_xlim[0] - xlim_range * 0.15, diff_xlim[1])
    else:
        current_xlim = ax_diff.get_xlim()
        ax_diff.set_xlim(current_xlim[0] - (current_xlim[1] - current_xlim[0]) * 0.25, current_xlim[1])

    # align x axes labels
    scatter_label_y = ax_scatter.xaxis.label.get_position()[1]
    diff_label_y = ax_diff.xaxis.label.get_position()[1]
    min_y = min(scatter_label_y, diff_label_y) - 0.04  # use lower position
    ax_scatter.xaxis.set_label_coords(0.5, min_y)
    ax_diff.xaxis.set_label_coords(0.5, min_y)

    sns.despine(ax=ax_scatter)
    sns.despine(ax=ax_diff, left=True)  # No left spine for diff plot

    # Save
    plt.savefig(os.path.join(save_path, f"{save_name}.{fmt}"),
                format=fmt, dpi=1000, bbox_inches="tight", pad_inches=0.01)
    plt.clf()
    plt.close()

    return


def c_v_ms_with_offDiag(preregistered_path, save_path):
    """
    Figure 1: c-v-ms correlation and off diagonals
    """
    long_data = pd.read_csv(os.path.join(preregistered_path, "c_v_ms", "c_v_ms_long.csv"))
    item_diff = pd.read_csv(os.path.join(preregistered_path, "c_v_ms", f"item_off_diagonal_differences.csv"))

    """
    Plot
    """
    # Prepare data
    df_pivot = long_data.pivot_table(index="Item", columns="Topic", values="Rating", aggfunc="mean").reset_index()
    df_pivot["Item"] = df_pivot["Item"].replace(survey_mapping.other_creatures_general_names)

    # compute colors for item_diff
    item_diff["color"] = np.select(
        [
            (item_diff["off_diagonal"]) & (item_diff["diff"] > 0),
            (item_diff["off_diagonal"]) & (item_diff["diff"] < 0)
        ],
        [1, -1],
        default=0
    )
    item_diff["Item"] = item_diff["Item"].replace(survey_mapping.other_creatures_general_names)

    # merge color into scatter df
    df_pivot = df_pivot.merge(item_diff[["Item", "color"]], on="Item", how="left")
    df_pivot["color"] = df_pivot["color"].fillna(0).astype(int)

    # colors
    colors_dict = {
        0: "#e5e5e5",
        1: YESNO_COLORS[survey_mapping.ANS_YES],
        -1: YESNO_COLORS[survey_mapping.ANS_NO]
    }

    # icon paths
    short_name_to_icon = {
        survey_mapping.other_creatures_general_names[k]: v
        for k, v in survey_mapping.other_creatures_icon_paths.items()
    }

    label_names = {1: "Does\nNot Have",
                   2: "Probably Doesn't Have",
                   3: "Probably Has",
                   4: "Has"}

    plot_scatter_with_differences(
        df_scatter=df_pivot,
        df_diff=item_diff,
        x_col="Consciousness", y_col="Moral Status", identity_col="Item",
        diff_value_col="diff", diff_category_col="color",
        group_col="color", group_colors=colors_dict,
        icon_paths=short_name_to_icon,
        save_path=save_path, save_name="fig1_combined",
        x_range=np.arange(1, 5, 1), y_range=np.arange(1, 5, 1),
        x_tick_labels=label_names,
        y_tick_labels=label_names, y_tick_rotation=90,
        icon_size=70, icon_max_width=70, diff_icon_max_width=40,
        alpha=0.95,
        border_width=0.5, border_color=(0, 0, 0),
        diag_line=True, show_annotations=False,
        se=True, se_col="se_diff", diff_alpha=1.0,
        size_inches_x=20, size_inches_y=14,
        scatter_width_ratio=0.77,  # wider scatter, narrower diff
        axis_label_fontsize=28, tick_fontsize=22,
        diff_title="", diff_title_fontsize=24,
        diff_icon_size=50,
        diff_label_fontsize=23, fmt="svg",
        diff_bar_height=0.5, diff_xlim=(-0.75, 0.75))
    return


def extra_panel(preregistered_path, save_path):
    question = survey_mapping.PRIOS_Q_NONCONS

    df = pd.read_csv(os.path.join(preregistered_path, "moral_consideration_prios", "moral_decisions_prios.csv"))
    # Add a dummy sample column for plot_pie (data is collapsed across samples)
    df["sample"] = "pre-registered"

    plot_pie(df=df, save_path=save_path, save_name="pie_nonconscious_moral", fmt="svg",
             stat_col=question,
             split_col="sample", split_order=["pre-registered"],
             stat_order=["Yes", "No"],
             color_palette=YESNO_COLORS,
             size_inches_x=8, size_inches_y=6,
             annot_fontsize=28,
             title_fontsize=26,
             legend_fontsize=22,
             shared_title="Should non-conscious creatures/systems\nbe considered in moral decisions?",
             show_subplot_titles=False,  # NEW — hide "All Samples" title
             show_legend=False,  # NEW — no legend
             show_slice_labels=True)
    return


def c_v_ms_scatter_offDiagPanel(preregistered_path, save_path):
    """
    Figure 1: c-v-ms correlation and off diagonals
    """
    long_data = pd.read_csv(os.path.join(preregistered_path, "c_v_ms", "c_v_ms_long.csv"))
    item_diff = pd.read_csv(os.path.join(preregistered_path, "c_v_ms", f"item_off_diagonal_differences.csv"))

    """
    Plot
    """
    df_pivot = long_data.pivot_table(index="Item", columns="Topic", values="Rating", aggfunc="mean").reset_index(drop=False, inplace=False)
    df_pivot["Item"] = df_pivot["Item"].replace(survey_mapping.other_creatures_general_names)

    # Compute color for item_diff
    item_diff["color"] = np.select(
        [
            (item_diff["off_diagonal"]) & (item_diff["diff"] > 0),
            (item_diff["off_diagonal"]) & (item_diff["diff"] < 0)
        ],
        [1, -1],
        default=0
    )

    # Merge color info into df_pivot
    item_diff_renamed = item_diff[["Item", "color"]].copy()
    item_diff_renamed["Item"] = item_diff_renamed["Item"].replace(survey_mapping.other_creatures_general_names)
    df_pivot = df_pivot.merge(item_diff_renamed, on="Item", how="left")
    df_pivot["color"] = df_pivot["color"].fillna(0).astype(int)

    # Icon paths
    short_name_to_icon = {
        survey_mapping.other_creatures_general_names[k]: v
        for k, v in survey_mapping.other_creatures_icon_paths.items()
    }

    # Colors dict - same as item_diff plot
    colors_dict = {
        0: "#e5e5e5",
        1: YESNO_COLORS[survey_mapping.ANS_YES],
        -1: YESNO_COLORS[survey_mapping.ANS_NO]
    }

    plot_scatter_icons(
        df=df_pivot,
        x_col="Consciousness",
        y_col="Moral Status",
        identity_col="Item",
        group_col="color",  # color by off-diagonal status
        group_order=[1, 0, -1],
        group_colors=colors_dict,
        icon_paths=short_name_to_icon,
        x_range=np.arange(1, 5, 1),
        y_range=np.arange(1, 5, 1),
        icon_size=52,
        icon_max_width=52,
        alpha=0.97,
        border_width=0.4,
        border_color=(0, 0, 0),
        vertical_jitter=0,
        horizontal_jitter=0,
        size_inches_x=18,
        size_inches_y=12,
        fmt="svg",
        save_path=save_path,
        save_name="correlation_c_ms_icons",
        legend_fontsize=18,
        diag_line=True,
        show_annotations=False,
        show_icon_legend=True,
        show_group_legend=False,
        icon_legend_size=40,
        icon_legend_max_width=30,
        icon_legend_fontsize=15,
        icon_legend_spacing=1.4,
        plot_left=0.06,
        plot_width=0.78,
        legend_left=0.79,
        legend_width=0.20)

    """
    Item Diff inset
    """
    # plot it (colors_dict already defined above)
    plotter.plot_item_differences_with_annotations(
        df=item_diff, id_col="Item", value_col="diff",
        category_col="color", color_map=colors_dict,
        save_path=save_path, save_name="item_off_diagonal_differences",
        se=True, se_col="se_diff", alpha=1.0, annotate=False,
        x_label="Difference", y_label="",
        x_tick_size=22, y_tick_size=22, annotate_fontsize=9,
        label_font_size=25,
        y_ticks_label_map=survey_mapping.other_creatures_general_names,
        plt_title="Off Diagonal", fmt="svg",
        size_inches_x=10,  size_inches_y_per_item=0.5)

    return


def plot_categorical_bars(df, category_col, value_col, categories_colors,
                          save_path, save_name, fmt="svg",
                          y_min=0, y_max=100, y_skip=10,
                          size_inches_x=22, size_inches_y=12,
                          order=None, text_width=12,
                          show_values=True, value_fontsize=26,
                          tick_fontsize=22, label_fontsize=26,
                          x_label="", y_label="Proportion (%)",
                          title="", title_fontsize=28,
                          bar_width=0.7, min_value_to_annotate=0,
                          # layered mode
                          layered=False, full_data_col=None, partial_data_col=None,
                          layered_alpha=0.4,
                          full_value_position="top", partial_value_position="middle"):
    """
    Plot categorical bars with optional layered mode and automatic text color.
    """

    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    fig, ax = plt.subplots(figsize=(size_inches_x, size_inches_y))

    # Determine the order of categories
    if order is None:
        categories = df[category_col].tolist()
    else:
        categories = order

    # Filter and reorder df
    df_plot = df.set_index(category_col).loc[categories].reset_index()

    x_positions = np.arange(len(categories))

    if layered and full_data_col and partial_data_col:
        # Layered bars: full (background) + partial (foreground)
        for i, (idx, row) in enumerate(df_plot.iterrows()):
            category = row[category_col]
            full_value = row[full_data_col]
            partial_value = row[partial_data_col]
            color = categories_colors.get(category, "#888888")

            # Full bar (background, lighter)
            ax.bar(x_positions[i], full_value, color=color, width=bar_width, alpha=layered_alpha, zorder=1)

            # Partial bar (foreground, solid)
            ax.bar(x_positions[i], partial_value, color=color, width=bar_width, alpha=1.0, zorder=2)

            if show_values:
                # Full bar annotation
                if full_value_position == "top":
                    full_y = full_value + 1
                    full_va = "bottom"
                    text_color = get_text_color("#FFFFFF")  # above bar, use dark text
                else:  # middle
                    full_y = full_value / 2
                    full_va = "center"
                    text_color = get_text_color(color)  # dark if dark, white if bright

                ax.text(x_positions[i], full_y, f"{full_value:.0f}%",
                        ha='center', va=full_va, fontsize=value_fontsize, color=text_color)

                # Partial bar annotation
                if partial_value >= min_value_to_annotate:
                    partial_y = partial_value / 2
                    partial_va = "center"
                    text_color = get_text_color(color)
                    ax.text(x_positions[i], partial_y, f"{partial_value:.0f}%",
                            ha='center', va=partial_va, fontsize=value_fontsize, color=text_color)
    else:
        # Standard single bars
        for i, (idx, row) in enumerate(df_plot.iterrows()):
            category = row[category_col]
            value = row[value_col]
            color = categories_colors.get(category, "#888888")

            ax.bar(x_positions[i], value, color=color, width=bar_width, alpha=1.0)

            if show_values and value > 0:
                text_color = get_text_color(color)
                ax.text(x_positions[i], value / 2, f"{value:.0f}%",
                        ha='center', va='center', fontsize=value_fontsize, color=text_color)

    # X-axis
    wrapped_labels = [textwrap.fill(str(cat), width=text_width) for cat in categories]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(wrapped_labels, fontsize=tick_fontsize)

    # Y-axis
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(np.arange(y_min, y_max + 1, y_skip))
    ax.set_yticklabels([f"{int(y)}" for y in np.arange(y_min, y_max + 1, y_skip)], fontsize=tick_fontsize)

    # Labels
    if x_label:
        ax.set_xlabel(x_label, fontsize=label_fontsize, labelpad=10)
    ax.set_ylabel(y_label, fontsize=label_fontsize, labelpad=10)

    # Title
    if title:
        ax.set_title(title, fontsize=title_fontsize, pad=15)

    sns.despine(ax=ax)

    # Save
    plt.savefig(os.path.join(save_path, f"{save_name}.{fmt}"),
                format=fmt, dpi=1000, bbox_inches='tight', pad_inches=0.01)
    plt.clf()
    plt.close()

    return


def figure_3(preregistered_path, save_path):
    df_path = os.path.join(preregistered_path, "moral_consideration_features", "important_features.csv")
    df = pd.read_csv(df_path)
    # plot the bar
    order = df.sort_values("proportion_all", ascending=False)["index"].tolist()

    plot_categorical_bars(
        df=df,
        category_col="index",
        value_col="proportion_all",
        categories_colors=analyze_survey.IMPORTANT_FEATURE_COLORS,
        save_path=save_path,
        save_name="important_features",
        fmt="svg",
        y_min=0, y_max=100, y_skip=20,
        size_inches_x=22, size_inches_y=10,
        order=order,
        text_width=14,
        show_values=True,
        value_fontsize=24,
        tick_fontsize=24,
        label_fontsize=26,
        x_label="",
        y_label="Proportion (%)",
        title="Features Important for Moral Consideration",
        title_fontsize=28,
        bar_width=0.7,
        layered=True,
        full_data_col="proportion_all",
        partial_data_col="proportion_one",
        layered_alpha=0.4,
        full_value_position="top",
        partial_value_position="middle", min_value_to_annotate=2)
    return


def plot_stacked_horizontal_bars(plot_data, colors, save_path, save_name,
                                 fmt="svg", title="", legend_labels=None,
                                 size_inches_x=20, size_inches_y=12,
                                 text_width=39, min_pct_to_label=4, bar_height=0.6,
                                 # font params
                                 title_fontsize=28, annot_fontsize=24,
                                 tick_fontsize=22, label_fontsize=26,
                                 legend_fontsize=20,
                                 # split mode
                                 split=False, yes_all_proportion=None, no_all_proportion=None,
                                 split_alpha=0.4,
                                 # layout
                                 subplot_hspace=0.3, x_label="Proportion (%)"):
    """
    Horizontal stacked proportion bars with current manuscript conventions.
    """

    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    num_plots = len(plot_data)

    fig, axs = plt.subplots(num_plots, 1, figsize=(size_inches_x, size_inches_y), sharex=True)
    axs = [axs] if num_plots == 1 else axs

    plt.subplots_adjust(hspace=subplot_hspace)

    for i, (col, data) in enumerate(plot_data):
        proportions = data["Proportion"]
        n = data["N"]
        ax = axs[i]

        bottom = 0

        if split and yes_all_proportion is not None and no_all_proportion is not None:
            # Split mode: show "to all" portions with reduced alpha
            no_to_all = 100 * no_all_proportion
            yes_to_all = 100 * yes_all_proportion
            no_to_this = proportions[0] - no_to_all
            yes_to_this = proportions[1] - yes_to_all

            segments = [
                ("No to all", no_to_all, colors[0], split_alpha),
                ("No", no_to_this, colors[0], 1.0),
                ("Yes", yes_to_this, colors[1], 1.0),
                ("Yes to all", yes_to_all, colors[1], split_alpha)
            ]
        else:
            # Standard mode
            segments = []
            for key in sorted(proportions.keys()):
                label = legend_labels.get(key, str(key)) if legend_labels else str(key)
                segments.append((label, proportions[key], colors[key], 1.0))

        # Plot segments
        for j, (label, proportion, color, alpha_val) in enumerate(segments):
            ax.barh(0, proportion, color=color, left=bottom, alpha=alpha_val, height=bar_height)

            # Annotate with automatic text color
            if proportion >= min_pct_to_label:
                text_color = get_text_color(color) if alpha_val >= 0.8 else "#333333"
                ax.text(bottom + proportion / 2, 0, f"{proportion:.0f}%",
                        ha='center', va='center', fontsize=annot_fontsize, color=text_color)

            bottom += proportion

        # Y-axis label (scenario name)
        wrapped_label = textwrap.fill(str(col), width=text_width)
        ax.set_yticks([0])
        ax.set_yticklabels([wrapped_label], fontsize=tick_fontsize)

        # X-axis
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])

        if i == num_plots - 1:  # only bottom subplot
            ax.set_xticklabels(['0', '25', '50', '75', '100'], fontsize=tick_fontsize)
        else:
            ax.set_xticklabels([])

        sns.despine(ax=ax)

    # X-axis label (shared)
    fig.text(0.5, 0.02, x_label, ha='center', fontsize=label_fontsize)

    # Title
    if title:
        fig.suptitle(title, fontsize=title_fontsize, y=0.98)

    # Legend with circles
    if legend_labels and not split:
        legend_handles = [
            Line2D([0], [0], marker='o', linestyle='None',
                   markerfacecolor=colors[k], markeredgecolor='none',
                   markersize=15, label=legend_labels[k])
            for k in sorted(legend_labels.keys())
        ]
        fig.legend(handles=legend_handles, loc='upper center',
                   bbox_to_anchor=(0.5, 0.94), ncol=len(legend_labels),
                   frameon=False, fontsize=legend_fontsize)
    elif split:
        # Legend for split mode: show full and discounted
        legend_handles = [
            Line2D([0], [0], marker='o', linestyle='None',
                   markerfacecolor=colors[0], markeredgecolor='none',
                   markersize=legend_fontsize * 0.8, alpha=1.0, label="No"),
            Line2D([0], [0], marker='o', linestyle='None',
                   markerfacecolor=colors[1], markeredgecolor='none',
                   markersize=legend_fontsize * 0.8, alpha=1.0, label="Yes"),
            Line2D([0], [0], marker='o', linestyle='None',
                   markerfacecolor='gray', markeredgecolor='none',
                   markersize=legend_fontsize * 0.8, alpha=split_alpha, label="Same response to all")
        ]
        fig.legend(handles=legend_handles, loc='upper center',
                   bbox_to_anchor=(0.5, 0.94), ncol=3,
                   frameon=False, fontsize=legend_fontsize)

    # Save
    plt.savefig(os.path.join(save_path, f"{save_name}.{fmt}"),
                format=fmt, dpi=1000, bbox_inches='tight', pad_inches=0.01)
    plt.clf()
    plt.close()

    return


def figure_2(preregistered_path, save_path):
    df_path = os.path.join(preregistered_path, "kill_for_test", "kill_to_pass_discounted.csv")
    df = pd.read_csv(df_path)

    # discounted horizontal bar plot
    plot_data = []
    for _, row in df.iterrows():
        label = row["0"]
        data_dict = eval(row["1"])
        plot_data.append((label, data_dict))

    # Sort by proportion of "No" (key 0) descending
    sorted_plot_data = sorted(plot_data, key=lambda x: x[1]["Proportion"][0], reverse=True)

    # Proportions for split mode (replace with actual values)
    yes_all_proportion = 0.10
    no_all_proportion = 0.25

    # Colors matching conventions
    rating_colors = {
        0: YESNO_COLORS[survey_mapping.ANS_NO],
        1: YESNO_COLORS[survey_mapping.ANS_YES]
    }
    rating_labels = {0: "No", 1: "Yes"}

    plot_stacked_horizontal_bars(
        plot_data=sorted_plot_data,
        colors=rating_colors,
        legend_labels=rating_labels,
        save_path=save_path,
        save_name="kill_to_pass_discounted",
        fmt="svg",
        title="Would you kill to pass the test?".title(),
        size_inches_x=20,
        size_inches_y=10,
        bar_height=0.4,
        text_width=40,
        min_pct_to_label=4,
        title_fontsize=36,
        annot_fontsize=33,
        tick_fontsize=33,
        label_fontsize=33,
        legend_fontsize=30,
        split=True,
        yes_all_proportion=yes_all_proportion,
        no_all_proportion=no_all_proportion,
        split_alpha=0.4,
        subplot_hspace=0.25,
        x_label="Proportion (%)"
    )
    return


def plot_cluster_preferences(df, cluster_col, question_cols, label_map, answer_labels,
                             cluster_colors, save_path, save_name,
                             fmt="svg", title="",
                             size_inches_x=18, size_inches_y=12,
                             # font params
                             title_fontsize=28, tick_fontsize=22,
                             label_fontsize=22, legend_fontsize=20,
                             legend_markersize=18, annot_fontsize=22,
                             # marker params
                             marker_size=20, marker_alpha=1.0, error_linewidth=2, cap_height=0.15,
                             # bar params
                             bar_height=0.6, bar_alpha=0.9,
                             # bar color mode: "majority" or "item"
                             bar_color_mode="item",
                             majority_colors=None,
                             item_colors=None,
                             show_percentages=True, min_pct_to_label=5,
                             # axis line params
                             show_axis_lines=False,
                             # center line params
                             center_line_width=1.0, center_line_style="--"):
    """
    Plot stacked horizontal bars with cluster centroids overlaid.
    """

    custom_params = {"axes.spines.right": False, "axes.spines.top": False,
                     "axes.spines.left": False, "axes.spines.bottom": False}
    sns.set_theme(style="ticks", rc=custom_params)

    fig, ax = plt.subplots(figsize=(size_inches_x, size_inches_y))

    n_questions = len(question_cols)
    y_positions = np.arange(n_questions)

    # Compute overall proportions for bars
    overall_proportions = df[question_cols].mean()

    # Compute cluster centroids and SEMs
    cluster_centroids = df.groupby(cluster_col)[question_cols].mean()
    cluster_sems = df.groupby(cluster_col)[question_cols].sem()

    n_clusters = len(cluster_centroids)

    # Draw stacked bars for each question
    for q_idx, q_col in enumerate(question_cols):
        prop_1 = overall_proportions[q_col]
        prop_0 = 1 - prop_1

        q_map = label_map.get(q_col, {})

        entity_0 = None
        entity_1 = None
        for ans, val in q_map.items():
            if val == 0:
                entity_0 = ans
            elif val == 1:
                entity_1 = ans

        if bar_color_mode == "item":
            color_0 = item_colors.get(entity_0, "#888888") if item_colors else "#888888"
            color_1 = item_colors.get(entity_1, "#888888") if item_colors else "#888888"
        else:
            if prop_1 > prop_0:
                color_1 = majority_colors.get("majority", "#102E4A")
                color_0 = majority_colors.get("minority", "#EDAE49")
            else:
                color_0 = majority_colors.get("majority", "#102E4A")
                color_1 = majority_colors.get("minority", "#EDAE49")

        left_width = 2 * prop_0
        right_width = 2 * prop_1
        split_point = -1 + left_width

        ax.barh(y_positions[q_idx], left_width, height=bar_height, left=-1,
                color=color_0, alpha=bar_alpha, zorder=1)

        ax.barh(y_positions[q_idx], right_width, height=bar_height, left=split_point,
                color=color_1, alpha=bar_alpha, zorder=1)

        if show_percentages:
            pct_0 = prop_0 * 100
            pct_1 = prop_1 * 100

            if pct_0 >= min_pct_to_label:
                text_color_0 = get_text_color(color_0)
                ax.text(-1 + left_width / 2, y_positions[q_idx], f"{pct_0:.0f}%",
                        ha='center', va='center', fontsize=annot_fontsize, color=text_color_0)

            if pct_1 >= min_pct_to_label:
                text_color_1 = get_text_color(color_1)
                ax.text(split_point + right_width / 2, y_positions[q_idx], f"{pct_1:.0f}%",
                        ha='center', va='center', fontsize=annot_fontsize, color=text_color_1)

    # Center line (preference = 0)
    ax.axvline(0, color='black', linewidth=center_line_width, linestyle=center_line_style, zorder=2)

    # Axis lines on left and right (optional)
    if show_axis_lines:
        ax.axvline(-1, color='black', linewidth=1, zorder=2)
        ax.axvline(1, color='black', linewidth=1, zorder=2)

    # Y-axis setup
    ax.set_yticks(y_positions)
    ax.set_yticklabels([""] * n_questions)

    # Add left/right labels with ticks for each question
    for q_idx, q_col in enumerate(question_cols):
        q_map = label_map.get(q_col, {})

        label_0 = None
        label_1 = None
        for ans, val in q_map.items():
            if val == 0:
                label_0 = answer_labels.get(ans, ans)
            elif val == 1:
                label_1 = answer_labels.get(ans, ans)

        # Title case the labels
        if label_0:
            label_0 = smart_title(label_0)
            # Left tick mark
            ax.plot([-1, -1.02], [y_positions[q_idx], y_positions[q_idx]],
                    color='black', linewidth=1, clip_on=False)
            ax.text(-1.05, y_positions[q_idx], label_0, va='center', ha='right',
                    fontsize=label_fontsize, color='black')
        if label_1:
            label_1 = smart_title(label_1)
            # Right tick mark
            ax.plot([1, 1.02], [y_positions[q_idx], y_positions[q_idx]],
                    color='black', linewidth=1, clip_on=False)
            ax.text(1.05, y_positions[q_idx], label_1, va='center', ha='left',
                    fontsize=label_fontsize, color='black')

    # X-axis
    x_padding = 0.08  # set x limits with padding to prevent clipping of the dots
    ax.set_xlim(-1 - x_padding, 1 + x_padding)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(['-1', '-0.5', '0', '0.5', '1'], fontsize=tick_fontsize)
    ax.set_xlabel("Preference", fontsize=tick_fontsize + 2, labelpad=10)

    ax.invert_yaxis()

    # Title
    if title:
        ax.set_title(title, fontsize=title_fontsize, pad=40)

    # Legend - clusters for dots (circles), horizontal below title
    legend_handles = [
        Line2D([0], [0], marker='o', linestyle='None',
               markerfacecolor=cluster_colors.get(c, "#888888"),
               markeredgecolor='black', markersize=legend_markersize,
               label=f"Cluster {c}")
        for c in sorted(cluster_centroids.index)
    ]

    ax.legend(handles=legend_handles, loc='upper center',
              bbox_to_anchor=(0.5, 1.06), ncol=n_clusters,
              frameon=False, fontsize=legend_fontsize)

    # Plot cluster dots on top
    for cluster_idx, cluster in enumerate(sorted(cluster_centroids.index)):
        centroids = cluster_centroids.loc[cluster]
        sems = cluster_sems.loc[cluster]

        for q_idx, q_col in enumerate(question_cols):
            mean_val = centroids[q_col] * 2 - 1
            sem_val = sems[q_col] * 2

            y_pos = y_positions[q_idx]
            color = cluster_colors.get(cluster, "#888888")

            # Calculate marker radius in data coordinates using axis transform
            # Get the display coordinates for two points
            display_coords = ax.transData.transform([[0, 0], [1, 0]])
            pixels_per_data_unit = display_coords[1, 0] - display_coords[0, 0]

            # marker_size is in points (1 point = 1/72 inch = fig.dpi/72 pixels)
            points_per_inch = 72
            marker_radius_pixels = (marker_size / 2) * (fig.dpi / points_per_inch)
            marker_radius_data = marker_radius_pixels / pixels_per_data_unit

            # Whiskers start at edge of marker, extend outward by SEM
            left_whisker_start = mean_val - marker_radius_data
            left_end = left_whisker_start - sem_val
            right_whisker_start = mean_val + marker_radius_data
            right_end = right_whisker_start + sem_val

            # Left whisker
            ax.plot([left_whisker_start, left_end], [y_pos, y_pos],
                    color='black', linewidth=error_linewidth, zorder=5, clip_on=False)
            # Left cap
            ax.plot([left_end, left_end], [y_pos - cap_height, y_pos + cap_height],
                    color='black', linewidth=error_linewidth, zorder=5, clip_on=False)

            # Right whisker
            ax.plot([right_whisker_start, right_end], [y_pos, y_pos],
                    color='black', linewidth=error_linewidth, zorder=5, clip_on=False)
            # Right cap
            ax.plot([right_end, right_end], [y_pos - cap_height, y_pos + cap_height],
                    color='black', linewidth=error_linewidth, zorder=5, clip_on=False)

            # Draw marker on top
            ax.plot(mean_val, y_pos, 'o',
                    color=color, markersize=marker_size, alpha=marker_alpha,
                    mec="black", mew=error_linewidth, zorder=6, clip_on=False)

    sns.despine(ax=ax, left=True, right=True, top=True, bottom=False)

    # Save
    plt.savefig(os.path.join(save_path, f"{save_name}.{fmt}"),
                format=fmt, dpi=1000, bbox_inches='tight', pad_inches=0.01)
    plt.clf()
    plt.close()

    return


def smart_title(text, preserve=None):
    """Title case that preserves specified acronyms/words."""
    if preserve is None:
        preserve = ["AI", "UWS", "DNA", "RNA", "PhD", "CEO", "USA", "UK"]

    words = text.split()
    result = []
    for word in words:
        if word.upper() in [p.upper() for p in preserve]:
            # Find the preserved version
            for p in preserve:
                if word.upper() == p.upper():
                    result.append(p)
                    break
        else:
            result.append(word.title())
    return " ".join(result)


def figure_4(preregistered_path, save_path):
    df_path = os.path.join(preregistered_path, "earth_danger", "earth_danger_clusters.csv")
    df = pd.read_csv(df_path)

    # plot the overlaid centroids
    question_cols = [c for c in df.columns if c not in ["response_id", "Cluster"]]

    for q_col in question_cols:
        print(f"\n{q_col}:")
        print(df.groupby("Cluster")[q_col].agg(['mean', 'std', 'sem', 'count']))

    # plot per MAJORITY VOTE
    plot_cluster_preferences(
        df=df,
        cluster_col="Cluster",
        question_cols=question_cols,
        label_map=survey_mapping.EARTH_DANGER_QA_MAP,
        answer_labels=survey_mapping.EARTH_DANGER_ANS_MAP,
        cluster_colors=EARTH_DANGER_CLUSTER_COLORS_MAJORITY,
        save_path=save_path,
        save_name="eid_preferences_perMajority",
        fmt="svg",
        title="Who Would You Save?",
        size_inches_x=18,
        size_inches_y=12,
        title_fontsize=28,
        tick_fontsize=22,
        label_fontsize=24,
        legend_fontsize=22,
        legend_markersize=20,
        annot_fontsize=24,
        marker_size=35,
        marker_alpha=0.9,
        error_linewidth=1,
        cap_height=0.15,
        bar_height=0.7,
        bar_alpha=1,
        bar_color_mode="majority",  # we can go bar_color_mode="item",
        majority_colors=MAJORITY_COLORS,  # and then do item_colors=EARTH_DANGER_COLOR_MAP,
        show_percentages=True,
        min_pct_to_label=5,
        show_axis_lines=False,  # no axis lines on sides
        center_line_width=1.0,
        center_line_style="--")

    return


def figure_1(preregistered_path, save_path):
    # option 1: c-v-ms correlation, with off diagonals already in the figure
    c_v_ms_with_offDiag(preregistered_path=preregistered_path, save_path=save_path)

    # option 2: just the c-v-ms correlation, with icons and off-diagonal coloring but w/o the off diagonal chart
    c_v_ms_colored_by_offDiag(preregistered_path=preregistered_path, save_path=save_path)

    # and have an extra panel of the off diagonals just in case
    extra_panel(preregistered_path=preregistered_path, save_path=save_path)

    # option 3: just the c-v-ms correlation, off-diagonal totally separate, scatter plot and not as icons
    c_v_ms_scatter_offDiagPanel(preregistered_path=preregistered_path, save_path=save_path)
    return


def manage_main_plotting(main_figs_path, preregistered_path):
    """
    Manage the plotting of the main, based on the outputs already existing for the manuscript without calcualting
    anything new.
    :param main_figs_path: where should the figures go
    :param preregistered_path: the path to the replication sample (what we report in the main
    """
    """
    Figure 1
    """
    figure_1(preregistered_path=preregistered_path, save_path=os.path.join(main_figs_path, "fig 1"))

    """
    Figure 2
    """
    figure_2(preregistered_path=preregistered_path, save_path=os.path.join(main_figs_path, "fig 2"))

    """
    Figure 3
    """
    figure_3(preregistered_path=preregistered_path, save_path=os.path.join(main_figs_path, "fig 3"))

    """
    Figure 4
    """
    figure_4(preregistered_path=preregistered_path, save_path=os.path.join(main_figs_path, "fig 4"))

    return


if __name__ == '__main__':
    """
    Step 1: prepare NHB MAIN figures
    """
    manage_main_plotting(
        main_figs_path=r"C..\main_figs",
        preregistered_path=r"..\replication")

    """
    Step 2: plotting FOR SUPPLEMENTARY!
    """
    # combine all samples
    exploratory_path = r"..\exploratory"
    preregistered_path = r"..\replication"
    followup_path = r"..\follow-up"
    save_path = r"..\supp_figs"
    # sub_df = unify_data_all(exploratory_path, preregistered_path, followup_path, save_path)
    sub_df = pd.read_csv(os.path.join(save_path, "sub_df.csv"))

    manage_plotting(sub_df=sub_df, save_path=save_path)







