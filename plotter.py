import os
import re
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
import geopandas as gpd
import math
import datamapplot
import geodatasets

# Modify the default font settings
mpl.rcParams['font.family'] = 'sans-serif'  # Change to serif fonts
mpl.rcParams['font.sans-serif'] = ['Verdana']  # Specify a serif font

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

nice_palette = ["#C17C74", "#BCAC9B", "#DDC9B4", "#2A3D45", "#7A6C5D"]

max_text_width = 20  # characters per line


def plotter_umap_embeddings(embeddings, topics, labels_dict, save_path, save_name, fmt="svg", dpi=1000,
                            dynamic_label_size=True, use_medoids=True, label_color_list=None,
                            size_inches_x=15, size_inches_y=15):

    string_labels = [labels_dict.get(topic, "Noise").split("Label for: ")[1].split(" (")[0].title() if topic != -1 else "Noise" for topic in topics]

    if label_color_list is not None:
        string_label_set = list(set(string_labels))
        label_color_map = {string_label_set[i]: label_color_list[i] for i in range(len(string_label_set))}
    else:
        label_color_map = None

    fig, ax = datamapplot.create_plot(data_map_coords=embeddings,
                                      labels=string_labels,
                                      label_font_size=15,
                                      title=f"Topic representations",
                                      sub_title=f"Representations on reduced 2D embeddings",
                                      label_wrap_width=20,
                                      use_medoids=use_medoids,  # Whether to use medoids instead of centroids to determine the "location" of the cluster
                                      dynamic_label_size=dynamic_label_size,
                                      dpi=dpi,
                                      label_color_map=label_color_map
                                      )

    # save plot
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(size_inches_x, size_inches_y)
    plt.savefig(os.path.join(save_path, f"{save_name}.{fmt}"), format=fmt, dpi=dpi,
                bbox_inches='tight', pad_inches=0.01)
    del figure
    plt.clf()
    plt.close()
    return


def diverging_palette(color_order, left, right):
    pal = sns.diverging_palette(left, right, as_cmap=True)
    num_colors = len(color_order)
    colors = [pal(i / (num_colors - 1)) for i in range(num_colors)]
    color_dict = {color_order[i]: colors[i] for i in range(num_colors)}
    return color_dict


def plot_raincloud_separate_samples(df, id_col, data_col_names, data_col_colors, save_path, save_name,
                                    x_title, x_name_dict, title,
                                    y_title, ymin, ymax, yskip, y_ticks=None, y_jitter=0,
                                    data_col_violin_left=None, violin_alpha=0.65, violin_width=0.5, group_spacing=0.5,
                                    marker_spread=0.1, marker_size=100, marker_alpha=0.25, fmt="svg",
                                    size_inches_x=15, size_inches_y=12):
    # ids
    ids = df[id_col].unique().tolist()
    # X axis params
    stim_xs = {item: idx * group_spacing for idx, item in enumerate(data_col_names)}
    scatter_x_dict = {id: {} for id in ids}
    scatter_y_dict = {id: {} for id in ids}

    for data_type in data_col_names:
        data_color = data_col_colors[data_type]
        y_values = df[data_type].dropna().tolist()  # drop nan - each column is its own group
        x_loc = stim_xs[data_type]

        # violin orientation
        if data_col_violin_left is None:
            left_flag = True
        else:
            left_flag = data_col_violin_left[data_type]

        # plot violin
        violin = plt.violinplot(y_values, positions=[x_loc], widths=violin_width, showmeans=True,
                                showextrema=False, showmedians=False)
        if left_flag:
            # make it a half-violin plot (only to the LEFT of center)
            for b in violin['bodies']:
                # get the center
                m = np.mean(b.get_paths()[0].vertices[:, 0])
                # modify the paths to not go further right than the center
                b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
                b.set_color(data_color)
                b.set_alpha(violin_alpha)
                b.set_edgecolor(data_color)
        else:
            # make it a half-violin plot (only to the RIGHT of center)
            for b in violin['bodies']:
                # get the center
                m = np.mean(b.get_paths()[0].vertices[:, 0])
                # modify the paths to not go further left than the center
                b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
                b.set_color(data_color)
                b.set_alpha(violin_alpha)
                b.set_edgecolor(data_color)

        # change the color of the mean lines (showmeans=True)
        violin['cmeans'].set_color("black")
        violin['cmeans'].set_linewidth(4)

        # control the length
        m = np.mean(violin['cmeans'].get_paths()[0].vertices[:, 0])
        if left_flag:
            violin['cmeans'].get_paths()[0].vertices[:, 0] = np.clip(violin['cmeans'].get_paths()[0].vertices[:, 0],
                                                                     -np.inf, m)
        else:
            violin['cmeans'].get_paths()[0].vertices[:, 0] = np.clip(violin['cmeans'].get_paths()[0].vertices[:, 0], m,
                                                                     np.inf)

        # now, scatter
        for id in ids:
            id_data = df[df[id_col] == id].reset_index(inplace=False, drop=True)
            id_y_loc = id_data[data_type][0]  # the first (and only) value here
            id_y_jitter = id_y_loc + np.random.uniform(-y_jitter, y_jitter, size=1)
            # introduce jitter to x_loc for this specific id (so that they won't all overlap)
            if left_flag:
                id_x_loc = (x_loc + marker_spread / 3.5) + (np.random.rand(1) * marker_spread)[0]
            else:
                id_x_loc = (x_loc - marker_spread / 3.5) - (np.random.rand(1) * marker_spread)[0]
            scatter_x_dict[id][data_type] = id_x_loc
            scatter_y_dict[id][data_type] = id_y_jitter
            plt.scatter(x=id_x_loc, y=id_y_jitter, marker="o", color=data_color, s=marker_size, alpha=marker_alpha,
                        edgecolor=data_color, zorder=2)

    # cosmetics
    plt.ylabel(y_title, fontsize=15, labelpad=8)
    if y_ticks is None:
        plt.yticks([y for y in np.arange(ymin, ymax, yskip)], fontsize=15)
    else:
        plt.yticks(ticks=[y for y in np.arange(ymin, ymax, yskip)], labels=[item for item in y_ticks], fontsize=15)
    plt.xlabel(x_title, fontsize=22, labelpad=10)
    plt.xticks(ticks=[(idx * group_spacing) for idx, item in enumerate(data_col_names)],
               labels=[x_name_dict[item] for item in data_col_names], fontsize=15)
    plt.title(title, fontsize=25)

    # save plot
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(size_inches_x, size_inches_y)
    plt.savefig(os.path.join(save_path, f"{save_name}.{fmt}"), format=fmt, dpi=1000, bbox_inches='tight',
                pad_inches=0.01)
    del figure
    plt.clf()
    plt.close()
    return


def plot_raincloud(df, id_col, data_col_names, data_col_colors, save_path, save_name,
                   x_title, x_name_dict, title,
                   y_title, ymin, ymax, yskip, y_ticks=None, y_jitter=0,
                   data_col_violin_left=None, violin_alpha=0.65, violin_width=0.5, group_spacing=0.5,
                   marker_spread=0.1, marker_size=100, marker_alpha=0.25, scatter_lines=True, fmt="svg",
                   size_inches_x=15, size_inches_y=12):
    # ids
    ids = df[id_col].unique().tolist()
    # X axis params
    stim_xs = {item: idx * group_spacing for idx, item in enumerate(data_col_names)}
    scatter_x_dict = {id: {} for id in ids}
    scatter_y_dict = {id: {} for id in ids}

    for data_type in data_col_names:
        data_color = data_col_colors[data_type]
        y_values = df[data_type].tolist()
        x_loc = stim_xs[data_type]

        # violin orientation
        if data_col_violin_left is None:
            left_flag = True
        else:
            left_flag = data_col_violin_left[data_type]

        # plot violin
        violin = plt.violinplot(y_values, positions=[x_loc], widths=violin_width, showmeans=True,
                                showextrema=False, showmedians=False)

        if left_flag:
            # make it a half-violin plot (only to the LEFT of center)
            for b in violin['bodies']:
                # get the center
                m = np.mean(b.get_paths()[0].vertices[:, 0])
                # modify the paths to not go further right than the center
                b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
                b.set_color(data_color)
                b.set_alpha(violin_alpha)
                b.set_edgecolor(data_color)
        else:
            # make it a half-violin plot (only to the RIGHT of center)
            for b in violin['bodies']:
                # get the center
                m = np.mean(b.get_paths()[0].vertices[:, 0])
                # modify the paths to not go further left than the center
                b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
                b.set_color(data_color)
                b.set_alpha(violin_alpha)
                b.set_edgecolor(data_color)

        # change the color of the mean lines (showmeans=True)
        violin['cmeans'].set_color("black")
        violin['cmeans'].set_linewidth(4)

        # control the length
        m = np.mean(violin['cmeans'].get_paths()[0].vertices[:, 0])
        if left_flag:
            violin['cmeans'].get_paths()[0].vertices[:, 0] = np.clip(violin['cmeans'].get_paths()[0].vertices[:, 0],
                                                                     -np.inf, m)
        else:
            violin['cmeans'].get_paths()[0].vertices[:, 0] = np.clip(violin['cmeans'].get_paths()[0].vertices[:, 0], m,
                                                                     np.inf)

        # now, scatter
        for id in ids:
            id_data = df[df[id_col] == id].reset_index(inplace=False, drop=True)
            id_y_loc = id_data[data_type][0]  # the first (and only) value here
            id_y_jitter = id_y_loc + np.random.uniform(-y_jitter, y_jitter, size=1)
            # introduce jitter to x_loc for this specific id (so that they won't all overlap)
            if left_flag:
                id_x_loc = (x_loc + marker_spread / 3.5) + (np.random.rand(1) * marker_spread)[0]
            else:
                id_x_loc = (x_loc - marker_spread / 3.5) - (np.random.rand(1) * marker_spread)[0]
            scatter_x_dict[id][data_type] = id_x_loc
            scatter_y_dict[id][data_type] = id_y_jitter
            plt.scatter(x=id_x_loc, y=id_y_jitter, marker="o", color=data_color, s=marker_size, alpha=marker_alpha,
                        edgecolor=data_color, zorder=2)

    # If we want to connect scatter dots, we should do it now
    if scatter_lines:
        for id in ids:
            y_values = [scatter_y_dict[id][col] for col in data_col_names]
            x_values = [scatter_x_dict[id][col] for col in data_col_names]
            plt.plot(x_values, y_values, color="lightgray", linewidth=0.65, alpha=0.65, zorder=1)

    # cosmetics
    plt.ylabel(y_title, fontsize=15, labelpad=8)
    if y_ticks is None:
        plt.yticks([y for y in np.arange(ymin, ymax, yskip)], fontsize=15)
    else:
        plt.yticks(ticks=[y for y in np.arange(ymin, ymax, yskip)], labels=y_ticks, fontsize=15)
    plt.xlabel(x_title, fontsize=22, labelpad=10)
    plt.xticks(ticks=[(idx * group_spacing) for idx, item in enumerate(data_col_names)],
               labels=[x_name_dict[item] for item in data_col_names], fontsize=15)
    plt.title(title, fontsize=25)

    # save plot
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(size_inches_x, size_inches_y)
    plt.savefig(os.path.join(save_path, f"{save_name}.{fmt}"), format=fmt, dpi=1000, bbox_inches='tight',
                pad_inches=0.01)
    del figure
    plt.clf()
    plt.close()
    return


def plot_pie(categories_names, categories_counts, title, save_path, save_name,
             fmt="svg", pie_direction=180, categories_colors=None, categories_labels=None,
             annot_groups=True, annot_group_selection=None, annot_props=True,
             legend=False, legend_order=None, legend_vertical=True,
             edge_color="white", rotate_labels=True, label_dist=1.01, props_in_legend=False):

    sns.set_style("ticks")
    sns.despine(right=True, top=True)
    plt.rcParams['font.family'] = "Calibri"

    if categories_colors is not None:
        try:
            categories_colors_list = [categories_colors[cat] for cat in categories_names]
        except Exception:
            print("Check manually")
    else:
        categories_colors_list = sns.color_palette("colorblind", len(categories_names))

    if categories_labels is not None:
        categories_labels_list = [categories_labels[cat] for cat in categories_names]
    else:
        categories_labels_list = categories_names

    """
    Pie direction: 
    - 90: largest category is at the top
    - 180: largest category is on the left
    - 270: largest category is at the bottom
    """
    startangle = pie_direction - (categories_counts[0] / sum(categories_counts)) * 180

    # pie plot
    fig, ax = plt.subplots(figsize=(20, 10))

    # proportions on top of each pie piece
    if annot_props:
        pctgs = "%1.0f%%"
    else:
        pctgs = ""

    # labels next to each pie piece
    if annot_groups:
        if annot_group_selection is None:
            label = categories_labels_list
        else:
            label = [l if l in annot_group_selection else None for l in categories_labels_list]
    else:
        label = [None] * len(categories_counts)

    # plot
    wedges, texts, autotexts = ax.pie(categories_counts, labels=label, autopct=pctgs,
                                      colors=categories_colors_list, rotatelabels=rotate_labels,
                                      labeldistance=label_dist, startangle=startangle, textprops={"fontsize": 18},
                                      wedgeprops={"edgecolor": edge_color})

    # add a title and subtitle
    ax.set_title(title.title(), fontsize=20, fontweight="normal")

    # legend
    if legend:
        if legend_vertical:
            location = "upper left"
            anchor = (1.02, 1)
            n_col = 1
        else:
            location = "lower center"
            anchor = (0.5, -0.1)
            n_col = len(categories_labels_list)

        total_count = sum(categories_counts)
        proportions = [f"{(count / total_count) * 100:.1f}%" for count in categories_counts]

        if props_in_legend:
            categories_labels_with_props = [f"{label} ({prop})" for label, prop in zip(categories_labels_list, proportions)]
        else:
            categories_labels_with_props = categories_labels_list

        if legend_order is None:
            ax.legend(wedges, categories_labels_with_props, loc=location, bbox_to_anchor=anchor, ncol=n_col, fontsize=18,
                      frameon=False)
        else:
            label_to_wedge_color = {label: (wedge, color, prop) for label, wedge, color, prop in
                                    zip(categories_labels_list, wedges, categories_colors_list, proportions)}

            ordered_wedges = [label_to_wedge_color[label][0] for label in legend_order if label in label_to_wedge_color]
            if props_in_legend:
                ordered_labels = [f"{label} ({label_to_wedge_color[label][2]})" for label in legend_order if
                                  label in label_to_wedge_color]
            else:
                ordered_labels = [label for label in legend_order if label in label_to_wedge_color]

            ax.legend(ordered_wedges, ordered_labels, loc=location, bbox_to_anchor=anchor, ncol=n_col, fontsize=17,
                      frameon=False)

    # save
    figure = plt.gcf()  # get current figure
    plt.savefig(os.path.join(save_path, f"{save_name}.{fmt}"), format=f"{fmt}", dpi=1000, bbox_inches='tight',
                pad_inches=0.01)
    plt.clf()
    plt.close()
    return


def plot_pca_scatter_2d(df, hue, title, save_path, save_name, pal=None, fmt="png", annotate=True, size=150):
    if pal is None:
        sns.scatterplot(x="PC1", y="PC2", data=df, hue=hue, s=size,
                        palette=sns.color_palette("hls", len(df[hue].unique().tolist())), legend=False)
    else:
        sns.scatterplot(x="PC1", y="PC2", data=df, hue=hue, s=size,
                        palette=pal, legend=False)
    if annotate:
        for item in df.index:
            plt.text(df.loc[item, 'PC1'], df.loc[item, 'PC2'], item)

    plt.title(title, fontsize=18)

    # axes
    plt.xlabel("Principal Component 1", fontsize=16)
    plt.ylabel("Principal Component 2", fontsize=16)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # save plot
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(15, 12)
    plt.savefig(os.path.join(save_path, f"{save_name}_PCA_result.{fmt}"), format=f"{fmt}", dpi=1000,
                bbox_inches='tight',
                pad_inches=0.01)
    del figure
    plt.clf()
    plt.close()
    return


def get_labels(label_dict, min_val=0, max_val=1):
    """
    * A HELPER FUNCTION TO plot_binary_preferences *
    Get the labels for "0" and "1" tags in a nested dictionary structure,
    where key=some category name, and value=a dictionary where keys are 0, 1 and values are their labels
    """
    label_0 = [key for key, value in label_dict.items() if value == min_val][0]
    label_1 = [key for key, value in label_dict.items() if value == max_val][0]
    return label_0, label_1


def plot_overlaid_preferences(all_preferences, all_sems, all_colors, labels, label_map, cluster_names,
                              binary, save_name, save_path, threshold=0, fmt="png", label_names_coding=None):
    """
    Creates a single plot overlaying centroids (preferences) and SEMs for all clusters.
    """

    plt.rcParams['font.family'] = "Calibri"

    fig, ax = plt.subplots(figsize=(16, 6))
    y_pos = np.arange(len(labels))

    # Map labels to readable names if label_map is provided
    display_labels = [label_map.get(label, label) for label in labels] if label_map else labels
    # use arbitratily the first label (if not binary) to get min and max
    if not binary:
        dummy_label = display_labels[0]
        min_max = sorted(list(dummy_label.values()))

    # Plot centroids for all clusters
    for cluster_idx, (preferences, sems, colors) in enumerate(zip(all_preferences, all_sems, all_colors)):
        for i, (mean, sem, color) in enumerate(zip(preferences, sems, colors)):
            ax.hlines(y=y_pos[i], xmin=(-1 if binary else min_max[0]), xmax=(1 if binary else min_max[1]),
                      color="lightgray", linestyle='--', linewidth=1)
            ax.errorbar(mean, y_pos[i], xerr=sem, fmt='o', color=color, markersize=15,
                        ecolor="black", elinewidth=2, capsize=4)

    ax.axvline(0 if binary else threshold, color='black', linewidth=1)
    ax.set_yticks(y_pos)
    plt.xticks(fontsize=18)
    ax.set_yticklabels(["" for _ in display_labels])
    ax.set_xlabel("No Preference" if binary else "Threshold", fontsize=22)
    ax.set_title("Overlaid Cluster Centroids", fontsize=22)
    ax.invert_yaxis()
    ax.set_xlim([-1, 1] if binary else [min_max[0], min_max[1]])

    for i, label in enumerate(display_labels):
        label_0, label_1 = get_labels(label_map[labels[i]]) if binary else get_labels(label_map[labels[i]],
                                                                                      min_val=1, max_val=4)
        if label_names_coding is not None:
            label_0 = label_names_coding[label_0]
            label_1 = label_names_coding[label_1]
        ax.text(-1.05 if binary else min_max[0] - 0.05, i, label_0, va='center', ha='right', fontsize=20, color='black')
        ax.text(1.05 if binary else min_max[1] + 0.05, i, label_1, va='center', ha='left', fontsize=20, color='black')

    # Add legend for clusters
    handles = [plt.Line2D([0], [0], color=colors[0], lw=4) for colors in all_colors]
    ax.legend(handles, cluster_names, loc="upper right", fontsize=20)

    # Save the figure
    figure = plt.gcf()
    figure.set_size_inches(15, 13)
    plt.savefig(os.path.join(save_path, f"{save_name}.{fmt}"), format=f"{fmt}", dpi=1000, bbox_inches='tight',
                pad_inches=0.01)
    del figure
    plt.clf()
    plt.close()
    return


def plot_nonbinary_preferences(means, sems, colors, labels, label_map, title, min, max, thresh,
                               save_name, save_path, fmt="png", label_names_coding=None):

    plt.rcParams['font.family'] = "Calibri"

    fig, ax = plt.subplots(figsize=(16, 6))
    y_pos = np.arange(len(means))

    # Plot dots with error bars and differential colors
    for i, (avg, sem, color) in enumerate(zip(means, sems, colors)):
        # plot a dashed line
        ax.hlines(y=y_pos[i], xmin=min - 0.05, xmax=max + 0.05, color="lightgray", linestyle='--', linewidth=1)
        # plot the actual data
        ax.errorbar(avg, y_pos[i], xerr=sem, fmt='o', color=color, markersize=8,
                    ecolor="black", elinewidth=2, capsize=4)

    ax.axvline(thresh, color="black", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(["" for i in labels])
    ax.set_xlabel("No Preference", fontsize=15)
    ax.set_title(title, fontsize=15)
    ax.invert_yaxis()
    ax.set_xlim([min, max])
    for i, label in enumerate(labels):
        label_0, label_1 = get_labels(label_map[label], min_val=min, max_val=max)
        if label_names_coding is not None:
            label_0 = label_names_coding[label_0]
            label_1 = label_names_coding[label_1]
        ax.text(min - 0.05, i, label_0, va='center', ha='right', fontsize=15, color='black')
        ax.text(max + 0.05, i, label_1, va='center', ha='left', fontsize=15, color='black')

    # save plot
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(15, 13)
    plt.savefig(os.path.join(save_path, f"{save_name}.{fmt}"), format=f"{fmt}", dpi=1000, bbox_inches='tight',
                pad_inches=0.01)
    del figure
    plt.clf()
    plt.close()
    return


def plot_binary_preferences(means, sems, colors, labels, label_map, title, save_name, save_path, fmt="png",
                            label_names_coding=None):
    plt.rcParams['font.family'] = "Calibri"
    fig, ax = plt.subplots(figsize=(16, 6))
    y_pos = np.arange(len(means))

    # ax.barh(y=y_pos, width=means, color=['#1f77b4' if val > 0 else '#ff7f0e' for val in means])  # if I want bars
    # ax.scatter(x=means, y=y_pos, color=['#1f77b4' if val > 0 else '#ff7f0e' for val in means], s=100)  # scatter w/o error bars

    # Plot dots with error bars and differential colors
    for i, (avg, sem, color) in enumerate(zip(means, sems, colors)):
        # plot a dashed line
        ax.hlines(y=y_pos[i], xmin=-1.05, xmax=1.05, color="lightgray", linestyle='--', linewidth=1)
        # plot the actual data
        ax.errorbar(avg, y_pos[i], xerr=sem, fmt='o', color=color, markersize=8,
                    ecolor="black", elinewidth=2, capsize=4)

    ax.axvline(0, color='black', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(["" for i in labels])
    ax.set_xlabel("No Preference", fontsize=20)
    ax.set_xticks(fontsize=20)
    ax.set_title(title, fontsize=20)
    ax.invert_yaxis()
    ax.set_xlim([-1, 1])
    for i, label in enumerate(labels):
        label_0, label_1 = get_labels(label_map[label])
        if label_names_coding is not None:
            label_0 = label_names_coding[label_0]
            label_1 = label_names_coding[label_1]
        ax.text(-1.05, i, label_0, va='center', ha='right', fontsize=18, color='black')
        ax.text(1.05, i, label_1, va='center', ha='left', fontsize=18, color='black')

    plt.legend(fontsize=18)
    # save plot
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(15, 12)
    plt.savefig(os.path.join(save_path, f"{save_name}.{fmt}"), format=f"{fmt}", dpi=1000, bbox_inches='tight',
                pad_inches=0.01)
    del figure
    plt.clf()
    plt.close()
    return


def plot_3d_scatter(x_col, y_col, z_col, data, color_map=None, c_values_col=None,
                    save=False, save_name=None, save_path=None, save_format="png"):
    x = data[x_col].tolist()
    y = data[y_col].tolist()
    z = data[z_col].tolist()

    fig = plt.figure(figsize=(10, 7))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    if color_map is not None:
        sc = ax.scatter(x, y, z,
                        s=40, c=data[c_values_col].tolist(), marker='o',
                        cmap=color_map)
    else:
        sc = ax.scatter(x, y, z,
                        s=40, marker='o', c="black")

    for i in range(0, len(x), 3):  # Step by 3 for triplets
        if i + 2 < len(x):  # Ensure we have a full triplet
            ax.plot([x[i], x[i + 1], x[i + 2]],
                    [y[i], y[i + 1], y[i + 2]],
                    [z[i], z[i + 1], z[i + 2]],
                    color="gray")

    if not save:
        c = 4
    #    plt.show()

    else:
        # save plot
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(15, 12)
        plt.savefig(os.path.join(save_path, f"{save_name}.{save_format}"), format=f"{save_format}",
                    dpi=1000, bbox_inches="tight", pad_inches=0.01)
        del figure
        plt.clf()
        plt.close()
    return


def plot_null_hist(observed_alpha, null_alphas, parameter_name_xlabel, save_path, save_name, fmt="svg",
                   observed_alpha_color="red", bins=30, alpha=0.7):
    sns.set_style("ticks")
    sns.despine(right=True, top=True)
    plt.rcParams['font.family'] = "Calibri"

    plt.figure(figsize=(10, 6))

    plt.hist(null_alphas, bins=bins, alpha=alpha, label="Null Alphas")
    plt.axvline(observed_alpha, color=observed_alpha_color, linestyle='--', linewidth=2,
                label=f"Observed α = {observed_alpha:.3f}")
    plt.xlabel(f"{parameter_name_xlabel} (null distribution)")
    plt.ylabel("Frequency")
    plt.title("")
    plt.legend()

    figure = plt.gcf()
    plt.savefig(os.path.join(save_path, f"{save_name}.{fmt}"), format=f"{fmt}", dpi=1000, bbox_inches="tight",
                pad_inches=0.01)
    del figure
    plt.clf()
    plt.close()
    return


def plot_histogram(df, category_col, data_col, save_path, save_name, format="svg", color_palette="Greens_d", color=None,
                   x_label="", y_label=""):

    sns.set_style("ticks")
    sns.despine(right=True, top=True)
    plt.rcParams['font.family'] = "Calibri"

    plt.figure(figsize=(20, 12))

    if color is None:  # use our default palette
        pal = sns.color_palette(color_palette, len(df))
        data = df.to_numpy()
        rank = data[:, 1].argsort().argsort()
        colors = np.array(pal)[rank]
        sns.barplot(x=data[:, 0], y=data[:, 1], palette=colors)
    else:  # use color
        sns.barplot(data=df, x=category_col, y=data_col, color=color)

    plt.xticks(fontsize=20)
    plt.xlabel(x_label, fontsize=25)
    plt.yticks(fontsize=22)
    plt.ylabel(y_label, fontsize=25)

    ax = plt.gca()  # get current axis
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # save plot
    figure = plt.gcf()
    plt.savefig(os.path.join(save_path, f"{save_name}.{format}"), format=f"{format}", dpi=1000, bbox_inches='tight',
                pad_inches=0.01)
    del figure
    plt.clf()
    plt.close()

    return


def plot_categorical_bars_layered(categories_prop_df, category_col, full_data_col, partial_data_col, categories_colors,
                                  save_path, save_name, fmt="svg", y_min=0, y_max=100, y_skip=10,
                                  inch_w=15, inch_h=12, order=None, annotate_bar=False, annot_font_color="white"):
    plt.figure(figsize=(8, 6))
    sns.set_style("ticks")
    plt.rcParams['font.family'] = "Calibri"

    # Determine the order of categories
    if order is None:
        categories = categories_prop_df[category_col].unique().tolist()
    else:
        categories = order

    # Plot the bars
    for category in categories:
        row = categories_prop_df[categories_prop_df[category_col] == category].iloc[0]  # category's data
        index = categories_prop_df.index[categories_prop_df[category_col] == category][0]  # category's index in df

        # full data
        full_bar = plt.bar(row[category_col], row[full_data_col], color=categories_colors[row[category_col]],
                            label="" if index == 0 else "", alpha=0.4)
        # partial data
        partial_bar = plt.bar(row[category_col], row[partial_data_col], color=categories_colors[row[category_col]],
                              label="" if index == 0 else "", alpha=1.0)

        if annotate_bar:
            # Annotate full bar
            for rect in full_bar:
                height = rect.get_height()
                if height > 0:
                    plt.text(rect.get_x() + rect.get_width() / 2, rect.get_height() / 2,
                             f"{height:.2f}%", ha='center', va='center', fontsize=20, color=annot_font_color)
            # Annotate partial bar
            for rect in partial_bar:
                height = rect.get_height()
                if height > 0:
                    partial_bar_y_position = rect.get_y() + height / 2  # Adjust y position based on the height
                    plt.text(rect.get_x() + rect.get_width() / 2, partial_bar_y_position,
                             f"{height:.2f}%", ha='center', va='center', fontsize=20, color=annot_font_color)

    plt.yticks([y for y in np.arange(y_min, y_max, y_skip)], fontsize=16)

    wrapped_labels = [textwrap.fill(label, width=10) for label in categories]
    plt.xticks(ticks=np.arange(len(wrapped_labels)), labels=wrapped_labels, fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlabel(category_col.title(), fontsize=20)
    plt.ylabel("Proportion", fontsize=25)

    ax = plt.gca()  # get current axis
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # save plot

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(inch_w, inch_h)
    plt.savefig(os.path.join(save_path, f"{save_name}.{fmt}"), format=f"{fmt}", dpi=1000, bbox_inches='tight',
                pad_inches=0.01)
    del figure
    plt.clf()
    plt.close()
    return


def plot_categorical_bars_hued(categories_prop_df, x_col, category_col, data_col, categories_colors, save_path,
                               save_name, fmt="svg", y_label="", y_min=0, y_max=100, y_skip=10, delete_y=True,
                               inch_w=15, inch_h=12, add_pcnt=True, order=None, x_label=None, x_rotation=0):
    plt.figure(figsize=(8, 6))
    sns.set_style("ticks")
    plt.rcParams['font.family'] = "Calibri"

    # Determine the order of categories
    if order is None:
        categories = categories_prop_df[x_col].unique().tolist()
    else:
        categories = order

    barplot = sns.barplot(x=x_col, y=data_col, hue=category_col, data=categories_prop_df, palette=categories_colors, order=categories)

    # add percentages on top of each bar
    if add_pcnt:
        for index, row in categories_prop_df.iterrows():
            # Get the X position for each bar, and the corresponding Y value (height of the bar)
            bar = barplot.patches[index]
            x_pos = bar.get_x() + bar.get_width() / 2  # X-coordinate of the center of the bar
            y_pos = bar.get_height()  # Y-coordinate at the top of the bar

            # Add text (percentage) above each bar
            barplot.text(x_pos, y_pos + 0.01, f"{row['Proportion']:.2f}%",
                    ha="center",
                    fontsize=14,
                    color="black")

    # now delete y-axis
    if delete_y:
        sns.despine(right=True, top=True, left=True)
        plt.ylabel("")
        plt.yticks([])
    else:
        plt.yticks([y for y in np.arange(y_min, y_max, y_skip)], fontsize=18)
        plt.ylabel(y_label, fontsize=22)

    # X axis and label
    wrapped_labels = [textwrap.fill(str(label), width=13) for label in categories]
    plt.xticks(ticks=np.arange(len(wrapped_labels)), labels=wrapped_labels, fontsize=18, rotation=x_rotation)
    if x_label is None:
        x_label = category_col.title()
    plt.xlabel(x_label, fontsize=22)

    # despine
    ax = plt.gca()  # get current axis
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # save plot
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(inch_w, inch_h)
    plt.savefig(os.path.join(save_path, f"{save_name}.{fmt}"), format=f"{fmt}", dpi=1000, bbox_inches='tight',
                pad_inches=0.01)
    del figure
    plt.clf()
    plt.close()

    return


def plot_categorical_bars(categories_prop_df, category_col, data_col, categories_colors,
                          save_path, save_name, fmt="svg", y_min=0, y_max=100, y_skip=10, delete_y=True,
                          inch_w=15, inch_h=12, add_pcnt=True, order=None, text_wrap_width=13,
                          x_label="", y_fontsize=28):

    plt.figure(figsize=(8, 6))
    sns.set_style("ticks")
    plt.rcParams['font.family'] = "Calibri"

    # Determine the order of categories
    if order is None:
        categories = categories_prop_df[category_col].unique().tolist()
    else:
        categories = order

    if categories_colors is None:
        barplot = sns.barplot(x=category_col, y=data_col, data=categories_prop_df)
    else:
        barplot = sns.barplot(x=category_col, y=data_col, data=categories_prop_df, palette=categories_colors)

    # add percentages on top of each bar
    if add_pcnt:
        for index, row in categories_prop_df.iterrows():
            barplot.text(
                index,  # X-coordinate (position of the bar)
                row[data_col] + 1,  # Y-coordinate (slightly above the bar)
                f"{row[data_col]:.0f}%",  # The text (percentage value)
                color="black",  # Text color
                ha="center",  # Horizontal alignment
                fontsize=28  # Font size
            )

    # now delete y-axis
    if delete_y:
        sns.despine(right=True, top=True, left=True)
        plt.ylabel("")
        plt.yticks([])
    else:
        plt.yticks([y for y in np.arange(y_min, y_max, y_skip)], fontsize=y_fontsize)
        plt.ylabel("Proportion", fontsize=y_fontsize)

    # X axis and label
    wrapped_labels = [textwrap.fill(str(label), width=text_wrap_width) for label in categories]
    plt.xticks(ticks=np.arange(len(wrapped_labels)), labels=wrapped_labels, fontsize=28)
    plt.xlabel(x_label, fontsize=28)
    ax = plt.gca()  # get current axis
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # save plot
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(inch_w, inch_h)
    plt.savefig(os.path.join(save_path, f"{save_name}.{fmt}"), format=f"{fmt}", dpi=1000, bbox_inches='tight', pad_inches=0.01)
    del figure
    plt.clf()
    plt.close()

    return


def plot_expertise_proportion_bars(df, x_axis_exp_col_name, x_label, cols, cols_colors, y_ticks,
                                   save_name, save_path, plt_title="", fmt="svg", x_map=None,
                                   plot_mean=False, stats_df=None, annotate_bar=False, annot_font_color="white"):
    sns.set_style("ticks")
    sns.despine(right=True, top=True)
    plt.rcParams['font.family'] = "Calibri"

    fig, ax = plt.subplots(figsize=(10, 6))

    if x_map is not None:
        # group and reorder according to x_map
        grouped = df.groupby(x_axis_exp_col_name).sum().reindex(x_map.keys())
        x_vals = [x_map[key] for key in grouped.index]
        bottom = np.zeros(len(grouped))
        for label in cols:
            values = grouped[label]
            ax.bar(x_vals, values, bottom=bottom, label=label, color=cols_colors[label], edgecolor='white')
            if annotate_bar:
                for idx, (x, val, bot) in enumerate(zip(x_vals, values, bottom)):
                    if val > 0:
                        ax.text(x, bot + val / 2, f"{val:.1f}%", ha='center', va='center', fontsize=12,
                                color=annot_font_color)
            bottom += values
        ax.set_xticks(list(x_map.values()))
        ax.set_xticklabels(list(x_map.keys()))
    else:
        # assume df[x] is numeric
        x_vals = df[x_axis_exp_col_name]
        bottom = np.zeros(len(df))
        for label in cols:
            values = df[label]
            ax.bar(x_vals, values, bottom=bottom, label=label, color=cols_colors[label], edgecolor='white')
            if annotate_bar:
                for idx, (x, val, bot) in enumerate(zip(x_vals, values, bottom)):
                    if val > 0:
                        ax.text(x, bot + val / 2, f"{val:.1f}%", ha='center', va='center', fontsize=10,
                                color=annot_font_color)
            bottom += values
        ax.set_xticks(sorted(x_vals.unique()))

    if plot_mean:
        if x_map is not None:
            stats_df['x_val'] = stats_df[x_axis_exp_col_name].map(x_map)
        else:
            stats_df['x_val'] = stats_df[x_axis_exp_col_name]

        for _, row in stats_df.iterrows():
            sem = row["std"] / np.sqrt(row["count"])
            ax.errorbar(
                x=row["x_val"], y=row["mean"], yerr=sem, fmt="o", color="black", capsize=5, label="")

    # labels and styling
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks)
    ax.set_ylabel("Proportion")
    ax.set_xlabel(x_label)
    ax.legend(title=plt_title)
    plt.title(plt_title)

    plt.tight_layout()
    sns.despine(right=True, top=True)
    plt.savefig(os.path.join(save_path, f"{save_name}.{fmt}"), format=fmt, dpi=1000, bbox_inches="tight", pad_inches=0.01)

    figure = plt.gcf()
    del figure
    plt.clf()
    plt.close()

    return


def plot_categorical_proportion_bar(categories_prop_df, category_col, data_col, categories_order, colors,
                                    save_path, save_name, title, legend=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    bottom = 0
    for i in range(len(categories_order)):
        category = categories_order[i]
        prop = categories_prop_df.loc[categories_prop_df[category_col] == category, data_col].tolist()[0]
        ax.barh(1, prop, color=colors[i], label=f"{categories_order[i]}", edgecolor='none', left=bottom,
                height=0.2)
        bottom += prop

    ax.tick_params(axis="x", labelsize=18)
    ax.set_yticks([0.85, 1, 1.45])
    ax.spines["left"].set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlim(0, 100)

    fig.text(0.5, 0.06, "Proportion of Responses (%)", ha="center", fontsize=18)
    fig.suptitle(f"{title}", fontsize=25, y=0.94)

    # legend
    if legend is not None:
        handles = [plt.Line2D([0], [0], color=colors[i], lw=10) for i in range(len(legend))]
        fig.legend(
            handles=handles,
            labels=legend,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.9),  # adjust y value to position the legend
            ncol=len(legend),  # arrange in a single row
            frameon=False,
            fontsize=18
        )

    # save plot
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(15, 12)
    plt.savefig(os.path.join(save_path, f"{save_name}.svg"), format="svg", dpi=1000, bbox_inches='tight',
                pad_inches=0.01)
    del figure
    plt.clf()
    plt.close()

    return


def plot_stacked_proportion_bars_in_a_batch(df, rating_col, item_cols, color_map, save_path, save_name, alpha_value=1.0,
                                            rating_label="", plot_title="", annotate=True,
                                            annot_font_size=15, annot_font_colors=None,
                                            ytick_visible=True, y_tick_rotation=0,
                                            text_width=max_text_width, fmt="svg"):

    sns.set_style("ticks")
    plt.rcParams['font.family'] = "Calibri"

    fig, axs = plt.subplots(len(item_cols), 1, figsize=(15, (len(list(color_map.keys())) + 1) * len(item_cols)), sharex=True)

    for i, item in enumerate(item_cols):
        a = axs[i]
        item_data = df[[rating_col, item]]
        item_data[f"{item} proportions"] = 100 * item_data[item] / item_data[item].sum()
        start = 0
        for index, row in item_data.iterrows():
            a.barh(0, row[f"{item} proportions"], left=start, height=0.8, color=color_map[row[rating_col]], label=item,
                   edgecolor="none", alpha=alpha_value)
            # annotate the bar with the proportion value
            if annotate:
                if annot_font_colors is None:
                    annot_font_colors = ["black" for i in range(len(list(color_map.keys())))]
                a.text(start + row[f"{item} proportions"] / 2, 0, f"{row[f'{item} proportions']:.2f}%",
                       ha='center', va='center', fontsize=annot_font_size, color=annot_font_colors[index])
            start += row[f"{item} proportions"]

        # customize each subplot
        a.tick_params(axis='x', labelsize=18)
        if ytick_visible:
            wrapped_col_name = textwrap.fill(str(item).title(), width=text_width)  # limit the text line width
            a.set_yticks(np.array([0]))
            a.set_yticklabels([wrapped_col_name], fontsize=22, rotation=y_tick_rotation, ha='center', va='center')
        else:
            a.set_yticklabels("")
            a.spines["left"].set_visible(False)
        a.set_xticks([0, 25, 50, 75, 100])
        a.set_xticklabels([0, 25, 50, 75, 100], fontsize=20)
        a.set_xlim(0, 100)  # Set x-axis limits from 0 to 100

        # despine
        a.spines["right"].set_visible(False)
        a.spines["top"].set_visible(False)


    # finalize figure
    fig.text(0.5, -0.05, "Proportion of Responses (%)", ha="center", fontsize=22)
    fig.suptitle(f"{plot_title}", fontsize=25, y=1.02)

    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[rating]) for rating in df[rating_col]]
    labels = [str(rating) for rating in df[rating_col]]
    fig.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1),  # adjust y value to position the legend
        ncol=len(handles),  # arrange in a single row
        frameon=False,
        fontsize=18
    )

    plt.tight_layout()
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(15, 12)
    plt.savefig(os.path.join(save_path, f"{save_name}.{fmt}"), format=f"{fmt}", dpi=1000, bbox_inches="tight",
                pad_inches=0.01)
    del figure
    plt.clf()
    plt.close()
    return


def plot_stacked_proportion_bars(plot_data, num_plots, colors, num_ratings, save_path, save_name, title,
                                 legend_labels=None, show_mean=True, sem_line=True, ytick_visible=True, y_title=None,
                                 default_ticks=True, text_width=max_text_width, fmt="png", annotate_bar=False,
                                 annot_font_color="white", split=False, relative=True,
                                 bar_relative=True, bar_range_min=1, bar_range_max=4, inches_w=18, inches_h=12,
                                 yes_all_proportion=None, no_all_proportion=None,
                                 punishment_alpha=0.6):
    """
    Plots horizontal stacked proportion bars for multiple items with optional error bars for mean ratings.

    :param plot_data: Dict, containing proportions, mean ratings, standard deviations, and N (# of people)
    :param num_plots: Number of plots to create (one per item).
    :param colors: List, colors for each rating category.
    :param num_ratings: Total number of rating categories.
    :param sem_line: Bool, optional. Whether to plot standard error lines. Defaults to True.
    :param ytick_visible: Whether to display y-axis tick labels. Defaults to True.
    :param text_width: Maximum width for wrapped text. Defaults to 30.
    """

    sns.set_style("ticks")
    plt.rcParams['font.family'] = "Calibri"
    fig, axs = plt.subplots(num_plots, 1, figsize=(15, (num_ratings + 1) * num_plots), sharex=True)

    # plot each column as a separate bar
    for i, (col, data) in enumerate(plot_data):
        proportions = data["Proportion"]  # between 0-100
        mean_rating = data["Mean"]
        std_dev = data["Std Dev"]
        n = data["N"]

        # validate proportions sum to ~100%
        if abs(sum(proportions.values()) - 100) > 1e-6:
            raise ValueError(f"Proportions for {col} do not sum to 100%: {sum(proportions.values()):.2f}%")

        # stacked bar plot
        bottom = 0  # left is bottom, as all bars will actually be horizontal

        if num_plots == 1:
            a = axs
        else:
            a = axs[i]

        if split:
            no_to_all = int(no_all_proportion * n)  # absolute number of people who answered 'No' to all
            no_to_this = int(proportions[0] * (n / 100) - no_to_all)  # subtract the fixed "No to all" portion
            yes_to_all = int(yes_all_proportion * n)  # absolute number of people who answered 'Yes' to all
            yes_to_this = int(proportions[1] * (n / 100) - yes_to_all)
            sorted_proportions = [
                ("No to all", 100 * no_to_all / n, colors[0], punishment_alpha),  # "No to all" with reduced alpha
                ("No", 100 * no_to_this / n, colors[0], 1.0),
                ("Yes", 100 * yes_to_this / n, colors[1], 1.0),
                ("Yes to all", 100 * yes_to_all / n, colors[1], punishment_alpha)  # "Yes to all" with reduced alpha
            ]
        else:
            if num_ratings == 2:
                sorted_proportions = [
                    ("No", proportions[0], colors[0], 1.0),
                    ("Yes", proportions[1], colors[1], 1.0)
                ]
            else:
                sorted_proportions = [(legend_labels[i], proportions[i + 1], colors[i], 1.0) for i in
                                      range(num_ratings)]

        if relative:
            for j, (label, proportion, color, alpha_value) in enumerate(sorted_proportions):
                a.barh(col, proportion, color=color, label=label, edgecolor='none', left=bottom, alpha=alpha_value)
                # annotate the bar with the proportion value
                if annotate_bar:
                    a.text(bottom + proportion / 2, 0, f"{proportion:.2f}%", ha='center', va='center', fontsize=21,
                           color=annot_font_color)
                bottom += proportion
        else:
            for j, (label, proportion, color, alpha_value) in enumerate(sorted_proportions):
                a.barh(0, proportion, color=color, label=label, edgecolor='none', left=bottom, alpha=alpha_value)
                # annotate the bar with the proportion value
                if annotate_bar:
                    a.text(bottom + proportion / 2, 0, f"{proportion:.2f}%", ha='center', va='center', fontsize=21,
                           color=annot_font_color)
                bottom += proportion

        if show_mean:
            # plot the mean rating as a dot
            if bar_relative:  # relative to actual proportions
                mean_position = (mean_rating / num_ratings) * 100
            else:  # ignoring them, scaling
                mean_position = bar_range_min + (mean_rating - 1) / (bar_range_max - bar_range_min) * 100
            a.plot(mean_position, 0, markersize=5, color="#333333")

            # annotate the mean rating
            a.text(
                mean_position + 0.6, 0,
                f"{mean_rating:.2f}",
                fontdict={"family": "sans-serif", "fontname": "Verdana", "fontsize": 16, "color": "black"},
                ha="left",  # Horizontal alignment
                va="center"  # Vertical alignment
            )

        if sem_line:
            # add standard error line
            if bar_relative:
                sem_position = (std_dev / np.sqrt(n)) * 100
            else:
                sem_position = (std_dev / np.sqrt(n)) * 100
            a.errorbar(mean_position, 0, xerr=sem_position, fmt='o', color="#333333",
                       ecolor="#333333", elinewidth=1, capsize=4, capthick=1, label="")

        # customize each subplot
        a.tick_params(axis='x', labelsize=18)
        if ytick_visible:
            wrapped_col_name = textwrap.fill(str(col), width=text_width)  # limit the text line width
            yticks = a.get_yticks()  # current y-tick positions
            if default_ticks:
                a.set_yticks(yticks)  # explicitly set them to satisfy FixedLocator
                a.set_yticklabels([wrapped_col_name] * len(yticks), fontsize=18)
            else:
                a.set_yticks(np.array([0]))
                a.set_yticklabels([wrapped_col_name], fontsize=18)
        else:
            a.set_yticklabels("")
            a.spines["left"].set_visible(False)
        a.set_xticks([0, 25, 50, 75, 100])
        a.set_xlim(0, 100)  # Set x-axis limits from 0 to 100
        # despine
        a.spines["right"].set_visible(False)
        a.spines["top"].set_visible(False)

    # finalize figure
    fig.text(0.5, 0.06, "Proportion of Responses (%)", ha="center", fontsize=18)
    fig.suptitle(f"{title}", fontsize=25, y=0.94)
    # y axis
    if y_title is not None:
        fig.text(0.06, 0.5, y_title, ha="center", va="center", rotation="vertical", fontsize=18)

    # legend
    if legend_labels is not None:
        handles = [plt.Line2D([0], [0], color=colors[i], lw=10) for i in range(len(legend_labels))]
        fig.legend(
            handles=handles,
            labels=legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.925),  # adjust y value to position the legend
            ncol=len(handles),  # arrange in a single row
            frameon=False,
            fontsize=18
        )

    # save plot
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(inches_w, inches_h)
    plt.savefig(os.path.join(save_path, f"{save_name}.{fmt}"), format=f"{fmt}", dpi=1000, bbox_inches='tight',
                pad_inches=0.01)
    del figure
    plt.clf()
    plt.close()

    return


def plot_scatter_xy_panel(df, identity_col, x_col, x_label, x_min, x_max, x_ticks, y_col, y_label, y_min, y_max,
                          y_ticks, ax, color_col=None, color_col_colors=None, palette_bounds=None, annotate_id=True,
                          title_text="", size=400, alpha=1, corr_line=False, diag_line=False,
                          individual_df=None, id_col=None, vertical_jitter=0, horizontal_jitter=0,
                          title_fontsize=14, axis_fontsize=12, hide_axes_names=False,
                          violins=False, violin_alpha=0.5, violin_color="gray"):

    sns.set_style("ticks")
    plt.rcParams['font.family'] = "Calibri"
    # use the provided axis for plotting
    plt.sca(ax)

    # dot color
    scaler = MinMaxScaler()
    df[["x_norm", "y_norm"]] = scaler.fit_transform(df[[x_col, y_col]])
    df["combined"] = df["x_norm"] + df["y_norm"]
    norm = plt.Normalize(vmin=df["combined"].min(), vmax=df["combined"].max())

    if palette_bounds is None:
        cmap = cm.get_cmap("viridis")
    else:
        cmap = LinearSegmentedColormap.from_list("custom", [palette_bounds[0], palette_bounds[1]])

    # individual participant lines
    if individual_df is not None:
        for id in individual_df[id_col].unique().tolist():
            id_df = individual_df.loc[individual_df[id_col] == id]
            id_df = id_df.sort_values(by=[id_col, x_col])
            sns.regplot(data=id_df, x=x_col, y=y_col, scatter=False, ci=None, order=1,
                        line_kws=dict(color="#B4B5BB", lw=0.15, linestyle="-"))

    # regression line
    if corr_line is True:
        sns.regplot(data=df, x=x_col, y=y_col, scatter=False, ci=None, order=1,  # linear
                    line_kws=dict(color="black", lw=1.50, linestyle="-"))

    if diag_line is True:
        start = math.floor(min(df[x_col].tolist()))
        end = math.ceil(max(df[x_col].tolist()))
        plt.plot([start, end], [start, end], color="gray", linestyle="dashed", linewidth=1.30, zorder=2)

    # jitter
    counts = df.groupby([x_col, y_col]).size().reset_index(name="counts")
    jitter_mask = counts[counts["counts"] > 1].set_index([x_col, y_col]).index
    # convert the jitter_mask to a set of tuples for efficient membership checking
    jitter_mask_set = set(jitter_mask)

    if vertical_jitter > 0 or horizontal_jitter > 0:
        df_copy = df.copy()  # untouched x_col, y_col
        for i in range(df.shape[0]):
            # create a tuple for the current point
            current_point = (df[x_col].iat[i], df[y_col].iat[i])
            # check if the current point is in the jitter_mask_set
            if current_point in jitter_mask_set:
                if vertical_jitter > 0:
                    # apply vertical jitter
                    df[y_col].iat[i] += np.random.uniform(-vertical_jitter, vertical_jitter)
                if horizontal_jitter > 0:
                    # apply horizontal jitter
                    df[x_col].iat[i] += np.random.uniform(-horizontal_jitter, horizontal_jitter)
    else:
        df_copy = df  # no need to create copies

    # scatter
    if color_col is None:  # cmap
        sns.scatterplot(data=df, x=x_col, y=y_col, cmap=cmap, norm=norm, c=df["combined"], s=size, alpha=alpha,
                        zorder=3, ax=ax)
    else:  # hue is the diff
        if color_col_colors is None:
            sns.scatterplot(data=df, x=x_col, y=y_col, cmap=cmap, norm=norm, hue=color_col, s=size, alpha=alpha,
                            zorder=3, ax=ax)
        else:
            sns.scatterplot(data=df, x=x_col, y=y_col, palette=color_col_colors, hue=color_col, s=size, alpha=alpha,
                            zorder=3, ax=ax)

    # distributions: we use data=df_copy --> to not jitter
    if violins:
        # X axis violin
        x_vals = df_copy[x_col].tolist()

        violin = plt.violinplot(x_vals, positions=[df[y_col].min()], vert=False,  # horizontal violin
                                showmeans=True, showextrema=False, showmedians=False)
        for b in violin['bodies']:
            b.set_alpha(violin_alpha)
            b.set_color(violin_color)
            # get the center
            m = np.mean(b.get_paths()[0].vertices[:, 1])  # 1 = get the center of the Y axis
            # modify the paths to not go further DOWN than the center
            b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], m, np.inf)
        # change color of mean line
        violin['cmeans'].set_color(violin_color)

        # Y axis violin
        y_vals = df_copy[y_col].tolist()
        violin = plt.violinplot(y_vals, positions=[df[x_col].min()], vert=True,  # vertical violin
                                showmeans=True, showextrema=False, showmedians=False)
        for b in violin['bodies']:
            b.set_alpha(violin_alpha)
            b.set_color(violin_color)
            # get the center
            m = np.mean(b.get_paths()[0].vertices[:, 0])  # 0 = get the center of the X axis
            # modify the paths to not go further RIGHT than the center
            b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
        # change color of mean line
        violin['cmeans'].set_color(violin_color)

    # annotate
    if annotate_id:
        for i in range(len(df)):
            ax.text(df[x_col][i], df[y_col][i] + 0.065, df[identity_col][i],
                    fontsize=10, ha="center")

    # Update limits based on jitter
    x_min_jittered = df[x_col].min() - horizontal_jitter
    x_max_jittered = df[x_col].max() + horizontal_jitter
    y_min_jittered = df[y_col].min() - vertical_jitter
    y_max_jittered = df[y_col].max() + vertical_jitter

    # adjust the limits based on the jitter
    ax.set_xlim(x_min_jittered, x_max_jittered)
    ax.set_ylim(y_min_jittered, y_max_jittered)

    # set titles and labels
    ax.set_title(title_text.title(), fontsize=title_fontsize)

    # set ticks for Y-axis
    ax.set_xticks(np.arange(x_min, x_max, x_ticks))
    ax.set_yticks(np.arange(y_min, y_max, y_ticks))
    ax.tick_params(axis="both", labelsize=axis_fontsize)

    # hide axis names based on parameter
    if not hide_axes_names:
        ax.set_xlabel(x_label.title(), fontsize=axis_fontsize)
        ax.set_ylabel(y_label.title(), fontsize=axis_fontsize)
    else:
        # Only label leftmost Y-axis and bottom X-axis based on index in the main plotting function
        ax.set_ylabel(y_label.title() if ax.get_subplotspec().is_first_col() else "", fontsize=axis_fontsize + 5)
        ax.set_xlabel(x_label.title() if ax.get_subplotspec().is_last_row() else "", fontsize=axis_fontsize + 5)
    return


def plot_multiple_scatter_xy(data, identity_col, x_col, y_col, x_label, y_label,
                             x_min, x_max, x_ticks, y_min, y_max, y_ticks, panel_per_col,
                             save_path, save_name, panel_order=None, color_col=None, color_col_colors=None,
                             palette_bounds=None, annotate_id=True, fmt="png",
                             size=400, alpha=1, corr_line=False, diag_line=False,
                             vertical_jitter=0, horizontal_jitter=0,
                             rows=4, cols=6, title_size=14, axis_size=12, hide_axes_names=False,
                             violins=False, violin_alpha=0.5, violin_color="gray"):

    # save name
    save_name = save_name if not violins else f"{save_name}_withViolins"

    # Create a figure with subplots
    sns.set_style("ticks")
    plt.rcParams['font.family'] = "Calibri"
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))  # Adjust size as needed
    axs = axs.flatten()  # Flatten the 2D array of axes to 1D for easy indexing

    # Define colors
    colors = [color_col_colors[0], color_col_colors[-1]] if color_col_colors else [None, None]

    # Loop through each unique item in custom order if provided
    if panel_order is None:
        items_to_plot = data[panel_per_col].unique().tolist()
    else:
        items_to_plot = panel_order
    for idx, item in enumerate(items_to_plot):
        data_item = data[data[panel_per_col] == item].drop(columns=[panel_per_col], inplace=False)
        # Call the existing plotting function with the specific axis for the subplot
        plot_scatter_xy_panel(df=data_item, identity_col=identity_col, x_col=x_col, x_label=x_label, x_min=x_min,
                              x_max=x_max,
                              x_ticks=x_ticks, y_col=y_col, y_label=y_label, y_min=y_min, y_max=y_max, y_ticks=y_ticks,
                              ax=axs[idx],  # Pass the current axis
                              color_col=color_col, color_col_colors=colors, palette_bounds=palette_bounds,
                              annotate_id=annotate_id,
                              title_text=item,  # Use the item name as the title
                              size=size, alpha=alpha, corr_line=corr_line, diag_line=diag_line,
                              vertical_jitter=vertical_jitter,
                              horizontal_jitter=horizontal_jitter,
                              title_fontsize=title_size, axis_fontsize=axis_size, hide_axes_names=hide_axes_names,
                              violins=violins, violin_alpha=violin_alpha, violin_color=violin_color)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{save_name}.{fmt}"), format=fmt, dpi=1000, bbox_inches="tight")
    plt.close(fig)
    return


def plot_categorical_scatter(df, x_col, xtick_labels, y_col, color, save_path, save_name,
                             fmt="svg", label_title="", label_x="", label_y="", size=100):
    plt.figure(figsize=(12, 5))
    sns.scatterplot(data=df, x=x_col, y=y_col, s=size, color=color)
    ticks = list(xtick_labels.keys())
    labels = [xtick_labels[tick] for tick in ticks]
    plt.xticks(ticks=ticks, labels=labels, rotation=90)
    plt.title(f"{label_title}")
    plt.xlabel(f"{label_x}")
    plt.ylabel(f"{label_y}")
    figure = plt.gcf()  # get current figure
    plt.savefig(os.path.join(save_path, f"{save_name}.{fmt}"), format=fmt, dpi=1000, bbox_inches="tight",
                pad_inches=0.01)
    del figure
    plt.clf()
    plt.close()
    return


def plot_categorical_multliscatter(df, x_col, xtick_labels, y_cols, colors, save_path, save_name, labels=None,
                                   fmt="svg", label_title="", label_x="", label_y="", y_ticks=None, size=100):
    plt.figure(figsize=(12, 5))
    for i in range(len(y_cols)):
        y_col = y_cols[i]
        color = colors[i]
        label = None if labels is None else labels[i]
        sns.scatterplot(data=df, x=x_col, y=y_col, s=size, color=color, label=label)
    x_ticks = list(xtick_labels.keys())
    x_labels = [xtick_labels[tick] for tick in x_ticks]
    plt.xticks(ticks=x_ticks, labels=x_labels, rotation=90)
    if y_ticks is not None:
        plt.yticks(ticks=y_ticks, labels=y_ticks)
    plt.title(f"{label_title}")
    plt.xlabel(f"{label_x}")
    plt.ylabel(f"{label_y}")
    plt.legend()
    figure = plt.gcf()  # get current figure
    plt.savefig(os.path.join(save_path, f"{save_name}.{fmt}"), format=fmt, dpi=1000, bbox_inches="tight",
                pad_inches=0.01)
    del figure
    plt.clf()
    plt.close()
    return


def plot_categorical_scatter_fullresponse(df, x_col, y_col, response_id_col, save_path, save_name, palette=None,
                                          s_scatter=100,
                                          show_means=True, color_mean="black", s_mean=100,
                                          hue_col=None, color_scatter="gray",
                                          show_se=True, show_sd=False,
                                          color_se="black", se_capsize=5, label_title="",
                                          label_x=None, label_y=None, x_jitter=0, y_jitter=0,
                                          y_ticks_list=None,
                                          lines=False, order=None, order_by="mean",
                                          scatter_alpha=0.6, fmt="svg"):

    plt.figure(figsize=(10, 6))
    plt.rcParams["font.family"] = "Calibri"
    sns.set_style("ticks")

    # Map x_col categories to numeric values for plotting
    if order is None:
        if order_by == "mean" or order_by == "std":
            stats = df.groupby(x_col)[y_col].agg([order_by]).reset_index()
        if order_by == "se":
            stats = df.groupby(x_col)[y_col].agg(["mean", "count", "std"]).reset_index()
            stats["se"] = stats["std"] / np.sqrt(stats["count"])
        stats = stats.sort_values(by=order_by, ascending=True, inplace=False)
        categories = stats[x_col].unique()
    else:
        categories = order

    cat_to_num = {cat: i for i, cat in enumerate(categories)}
    df["x_numeric"] = df[x_col].map(cat_to_num)

    # Apply jitter
    jittered_x = df["x_numeric"] + np.random.uniform(-x_jitter, x_jitter, size=len(df))
    jittered_y = df[y_col] + np.random.uniform(-y_jitter, y_jitter, size=len(df))

    if isinstance(color_scatter, list):
        # Map colors based on y value using the provided color list
        color_map = {y_val: color for y_val, color in zip(sorted(df[y_col].unique()), color_scatter)}
        colors = df[y_col].map(color_map)
    else:
        # If not a list, default to 'color_scatter' or use hue_col/palette
        if hue_col is None:
            colors = color_scatter
        else:
            if palette is None:
                sns.scatterplot(x=jittered_x, y=jittered_y, hue=df[hue_col], alpha=scatter_alpha, s=s_scatter)
            else:
                sns.scatterplot(x=jittered_x, y=jittered_y, hue=df[hue_col], palette=palette, alpha=scatter_alpha,
                                s=s_scatter)

        # Scatter plot with the selected colors
    if not isinstance(color_scatter, list) or len(color_scatter) != len(df):
        plt.scatter(jittered_x, jittered_y, color=colors, alpha=scatter_alpha, s=s_scatter)
    else:
        plt.scatter(jittered_x, jittered_y, color=colors, alpha=scatter_alpha, s=s_scatter)

    if lines:
        for _, group in df.groupby(response_id_col):
            xs = group[x_col].map(cat_to_num)
            ys = group[y_col]
            plt.plot(xs, ys, color="gray", alpha=0.3, linewidth=0.7)

    if show_means or show_se or show_sd:
        stats = df.groupby(x_col)[y_col].agg(["mean", "count", "std"]).reset_index()
        stats["se"] = stats["std"] / np.sqrt(stats["count"])
        stats["x_numeric"] = stats[x_col].map(cat_to_num)

        if show_means:
            sns.scatterplot(x=stats["x_numeric"], y=stats["mean"], color=color_mean, s=s_mean, label="")

        if show_se:
            plt.errorbar(x=stats["x_numeric"], y=stats["mean"], yerr=stats["se"],
                         fmt="none", ecolor=color_se, capsize=se_capsize)

        if show_sd:
            plt.errorbar(x=stats["x_numeric"], y=stats["mean"], yerr=stats["std"],
                         fmt="none", ecolor=color_se, capsize=se_capsize)

    # Set custom x-axis ticks
    plt.xticks(ticks=list(cat_to_num.values()), labels=list(cat_to_num.keys()), rotation=90, fontsize=18)
    if y_ticks_list is None:
        plt.yticks(fontsize=18)
    else:
        plt.yticks(y_ticks_list, y_ticks_list, fontsize=18)

    if label_x is None:
        plt.xlabel(x_col)
    else:
        plt.xlabel(f"{label_x}")
    if label_y is None:
        plt.ylabel(y_col)
    else:
        plt.ylabel(f"{label_y}")
    plt.title(f"{label_title}")

    sns.despine(right=True, top=True)
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(18, 12)
    plt.savefig(os.path.join(save_path, f"{save_name}.{fmt}"), format=fmt, dpi=1000, bbox_inches="tight",
                pad_inches=0.01)
    del figure
    plt.clf()
    plt.close()
    return


def plot_scatter_xy(df, identity_col, x_col, x_label, x_min, x_max, x_ticks, y_col, y_label, y_min, y_max, y_ticks,
                    save_path, save_name, color_col=None, color_col_colors=None, palette_bounds=None, annotate_id=True,
                    title_text="", fmt="png", size=600, alpha=1, corr_line=False, diag_line=False, individual_df=None,
                    id_col=None, vertical_jitter=0, horizontal_jitter=0):

    plt.figure(figsize=(8, 6))
    plt.rcParams["font.family"] = "Calibri"
    sns.set_style("ticks")

    # dot color
    scaler = MinMaxScaler()
    df[["x_norm", "y_norm"]] = scaler.fit_transform(df[[x_col, y_col]])
    df["combined"] = df["x_norm"] + df["y_norm"]
    norm = plt.Normalize(vmin=df["combined"].min(), vmax=df["combined"].max())

    if palette_bounds is None:
        cmap = cm.get_cmap("viridis")
    else:
        cmap = LinearSegmentedColormap.from_list("custom", [palette_bounds[0], palette_bounds[1]])

    # individual participant lines
    if individual_df is not None:
        for id in individual_df[id_col].unique().tolist():
            id_df = individual_df.loc[individual_df[id_col] == id]
            id_df = id_df.sort_values(by=[id_col, x_col])
            sns.regplot(data=id_df, x=x_col, y=y_col, scatter=False, ci=None, order=1,
                        line_kws=dict(color="#B4B5BB", lw=0.15, linestyle="-"))

    # regression line
    if corr_line is True:
        sns.regplot(data=df, x=x_col, y=y_col, scatter=False, ci=None, order=1,  # linear
                    line_kws=dict(color="black", lw=1.25, linestyle="--"))

    if diag_line is True:
        start = math.floor(min(df[x_col].tolist()))
        end = math.ceil(max(df[x_col].tolist()))
        # draw a line from (start, start) to (end, end)
        plt.plot([start, end], [start, end], color="#1F2041", linestyle="dashed", linewidth=1.25, zorder=2)

    """
    Jitter
    """
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

    # scatter plot
    if color_col is None:  # cmap
        sns.scatterplot(data=df, x=x_col, y=y_col, cmap=cmap, norm=norm, c=df["combined"], s=size, alpha=alpha,
                        zorder=3)
    else:  # hue is the diff
        if color_col_colors is None:
            sns.scatterplot(data=df, x=x_col, y=y_col, cmap=cmap, norm=norm, hue=color_col, s=size, alpha=alpha, zorder=3)
        else:
            sns.scatterplot(data=df, x=x_col, y=y_col, palette=color_col_colors, hue=color_col, s=size, alpha=alpha,
                            zorder=3)

    # annotate
    if annotate_id:
        for i in range(len(df)):
            plt.text(df[x_col][i], df[y_col][i] + 0.065, df[identity_col][i], fontsize=20, ha="center")

    # update limits based on jitter
    x_min_jittered = df[x_col].min() - horizontal_jitter
    x_max_jittered = df[x_col].max() + horizontal_jitter
    y_min_jittered = df[y_col].min() - vertical_jitter
    y_max_jittered = df[y_col].max() + vertical_jitter
    # adjust the limits based on the jitter
    plt.xlim(x_min_jittered, x_max_jittered)
    plt.ylim(y_min_jittered, y_max_jittered)

    # titles etc
    plt.yticks(np.arange(y_min, y_max + (0.05 * y_ticks), y_ticks), fontsize=22)
    plt.xticks(np.arange(x_min, x_max + (0.05 * x_ticks), x_ticks), fontsize=22)
    plt.xlim([x_min, x_max + (0.05 * y_ticks)])
    plt.ylim([y_min, y_max + (0.05 * y_ticks)])
    plt.xlabel(x_label.title(), fontsize=25)
    plt.ylabel(y_label.title(), fontsize=25)
    plt.title(f"{title_text.title()}", fontsize=16)

    # save plot
    sns.despine(right=True, top=True)
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(18, 12)
    plt.savefig(os.path.join(save_path, f"{save_name}.{fmt}"), format=fmt, dpi=1000, bbox_inches="tight",
                pad_inches=0.01)
    del figure
    plt.clf()
    plt.close()

    return


def plot_scatter(df, data_col, category_col, category_color_dict, category_order, title_text, x_label, y_label,
                 save_path, save_name, y_min, y_max, y_skip, vertical_jitter=0, horizontal_jitter=0.2):

    # calculate means and standard deviations for each category
    category_means = df.groupby(category_col)[data_col].mean()
    category_stds = df.groupby(category_col)[data_col].std()
    category_sems = df.groupby(category_col)[data_col].apply(lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))

    if vertical_jitter > 0:
        np.random.seed(42)
        df["jittered"] = df[data_col] + np.random.uniform(-vertical_jitter, vertical_jitter, size=len(df))
    else:
        df["jittered"] = df[data_col]

    plt.figure(figsize=(8, 6))
    sns.set_style("ticks")
    plt.rcParams['font.family'] = "Calibri"

    for category in category_order:
        sns.stripplot(x=category_col, y="jittered", data=df[df[category_col] == category],
                      jitter=horizontal_jitter, size=15, color=category_color_dict[category], alpha=0.8, zorder=1)

    # Plot means with standard deviations manually
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*All-NaN axis encountered.*")
    for i, category in enumerate(category_order):
        try:
            mean = category_means[category]
            std = category_stds[category]
            sem = category_sems[category]
        except Exception:  # this category does not exist (no one answered it)
            mean = np.nan
            std = np.nan
            sem = np.nan

        plt.plot(i, mean, "o", color="black", markersize=10, label=f"Mean {category}", zorder=2)
        # yerr=sem or yerr=std, whichever we prefer
        plt.errorbar(i, mean, yerr=sem, fmt="o", color="black", capsize=7, zorder=2)

    if y_min is None:
        y_min = df[data_col].min()
    if y_max is None:
        y_max = df[data_col].max() + 1
    if y_skip is None:
        y_skip = 1
    plt.yticks(np.arange(y_min, y_max + 1 / y_skip, y_skip), fontsize=16)
    plt.xticks(fontsize=16)
    plt.xlabel(x_label.title(), fontsize=18)
    plt.ylabel(y_label.title(), fontsize=18)
    plt.title(f"{title_text.title()}", fontsize=16)

    # save plot
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(16, 12)
    plt.savefig(os.path.join(save_path, f"{save_name}.png"), format="png", dpi=1000, bbox_inches='tight',
                pad_inches=0.01)
    del figure
    plt.clf()
    plt.close()

    return


def plot_world_map_proportion(country_proportions_df, data_column, save_path, save_name="proportion_by_country",
                              fmt="svg"):
    """
    :param country_proportions_df:
    :param data_column:
    :param save_path:
    :return:
    """

    """
    To load the world map I use here: (even though it's on the git repo as well)
    - Go to: https://www.naturalearthdata.com/downloads/110m-cultural-vectors/
    - Download the "Admin 0 – Countries" zip file
    - Save it on your computer and extract it
    - Then, map_shapfile_path should be the path to the '.shp' file in the unzipped folder
    
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    map_shapfile_path = os.path.join(dir_path, r"ne_110m_admin_0_countries\ne_110m_admin_0_countries.shp")

    world = gpd.read_file(map_shapfile_path)
    world = world[world["CONTINENT"] != "Antarctica"]  # remove Antarctica
    world = world[world["CONTINENT"] != "Seven seas (open ocean)"]  # remove the seven seas
    world.loc[world["NAME"] == "Russia", "CONTINENT"] = "Eastern Europe"  # Russia is otherwise clustered with the whole of Europe

    """
    Colormaps
    dark=0.1 is darker than 0.2, light=0.95 is lighter than 0.6, gamma<1 lighter in the middle, gamma=1 linear
    """
    #warm_brown_cmap = sns.color_palette("ch:start=0.5,rot=0.3", as_cmap=True)
    warm_brown_cmap = sns.color_palette("ch:start=0.5,rot=0.3,dark=0.1,light=0.8,gamma=1", as_cmap=True)

    """
    Plot by country
    """

    df = country_proportions_df.rename(columns={data_column: "NAME"})
    # match the country names to the ones appearing in the world map we loaded
    df["NAME"].replace({"United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
                        "Viet Nam": "Vietnam",
                        "Czech Republic": "Czechia"},
                       inplace=True)

    new_world = world.merge(df, on="NAME", how="left")

    # ensure countries are aligned
    world_with_props = new_world[~new_world["proportion"].isna()].reset_index(inplace=False, drop=True)
    diff = world_with_props.shape[0] - df.shape[0]
    if diff < 0:  # we have countries with proportions that don't exist in the world map
        df_props_countries = df["NAME"].unique().tolist()
        world_countries = world_with_props["NAME"].unique().tolist()
        diff_lst = list(set(df_props_countries) - set(world_countries))  # the missing countries
        if "Singapore" in diff_lst:  # the world map doesn't contain Singapore for some reason
            df["NAME"].replace({"Singapore": "Malaysia"}, inplace=True)
            new_world = world.merge(df, on="NAME", how="left")
            diff_lst.remove("Singapore")
        print(f"Mismatching countries: {diff_lst}")

    fig, ax = plt.subplots(1, 1, figsize=(18, 14))
    new_world.boundary.plot(ax=ax, linewidth=1, edgecolor="#050b0c")
    new_world.plot(column="proportion", ax=ax, legend=True,
                   cmap=warm_brown_cmap,
                   missing_kwds={"color": "white"},
                   legend_kwds={'label': "Proportion by Country",
                                'orientation': "horizontal",
                                'shrink': 0.4})

    ax.axis('off')

    # save
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(15, 12)
    plt.savefig(os.path.join(save_path, f"{save_name}_country.{fmt}"), format=f"{fmt}", dpi=1000, bbox_inches='tight',
                pad_inches=0.01)
    del figure
    plt.clf()
    plt.close()

    """
    Plot by continent
    """

    world_continent = new_world.loc[:, ["NAME", "CONTINENT"]]
    df_with_continent = df.merge(world_continent, on="NAME", how="left")

    # calculate continent proportions
    continent_proportions = df_with_continent["CONTINENT"].value_counts(normalize=True)
    continent_df = pd.DataFrame(continent_proportions).reset_index()
    continent_df.columns = ["CONTINENT", "continent_proportion"]
    continent_df["continent_proportion"] = continent_df["continent_proportion"] * 100
    new_world = new_world.merge(continent_df, on="CONTINENT", how="left")  # add proportion data to world map

    fig, ax = plt.subplots(1, 1, figsize=(18, 14))
    new_world.boundary.plot(ax=ax, linewidth=1, edgecolor="#050b0c")
    new_world.plot(column="continent_proportion", ax=ax, legend=True,
                   cmap=warm_brown_cmap,
                   missing_kwds={"color": "white"},
                   legend_kwds={'label': "Proportion by Continent",
                                'orientation': "horizontal",
                                'shrink': 0.4})
    ax.axis('off')

    # save
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(15, 12)
    plt.savefig(os.path.join(save_path, f"{save_name}_continent.{fmt}"), format=f"{fmt}", dpi=1000, bbox_inches='tight',
                pad_inches=0.01)
    del figure
    plt.clf()
    plt.close()

    """
    Save df with data
    """
    new_world_save = new_world.loc[:, ["NAME", "proportion", "CONTINENT", "continent_proportion"]]
    new_world_save.to_csv(os.path.join(save_path, f"{save_name}.csv"), index=False)

    return


def plot_density(df, x_col, x_col_name, hue_col, hue_col_name, save_name, save_path, format="png", pal="crest", a=0.3,
                 xskip=1):
    ax = sns.kdeplot(data=df, x=x_col, hue=hue_col, fill=True, common_norm=False, palette=pal, alpha=a, linewidth=0)

    # axes
    plt.ylabel("Density", fontsize=17, labelpad=10)
    plt.yticks([y for y in np.arange(0, 1 + (0.1 / 2), 0.1)], fontsize=15)
    plt.xlabel(x_col_name.title(), fontsize=17, labelpad=10)
    plt.xticks([x for x in np.arange(df[x_col].min(), df[x_col].max() + (xskip / 2), xskip)], fontsize=15)

    # legend
    labels = sorted([int(n) for n in df[hue_col].unique().tolist()])
    colors = [plt.Line2D([0], [0], color=sns.color_palette(pal, len(labels))[i], lw=10, alpha=a + 0.1) for i in
              range(len(labels))]
    ax.legend(colors, labels, title=hue_col_name.title(), fontsize=15, title_fontsize=18, ncol=len(labels),
              frameon=False,
              loc="upper center")

    # save
    figure = plt.gcf()  # get current figure
    sns.set_style("ticks")
    sns.despine(right=True, top=True)
    plt.rcParams["font.family"] = "Calibri"
    figure.set_size_inches(18, 12)
    plt.savefig(os.path.join(save_path, f"density_{save_name}.{format}"), format=format, dpi=1000, bbox_inches='tight',
                pad_inches=0.01)
    del figure
    plt.clf()
    plt.close()
    return
