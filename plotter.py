import os
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
import geodatasets

# Modify the default font settings
mpl.rcParams['font.family'] = 'sans-serif'  # Change to serif fonts
mpl.rcParams['font.sans-serif'] = ['Verdana']  # Specify a serif font

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

nice_palette = ["#C17C74", "#BCAC9B", "#DDC9B4", "#2A3D45", "#7A6C5D"]

max_text_width = 20  # characters per line


def diverging_palette(color_order, left, right):
    pal = sns.diverging_palette(left, right, as_cmap=True)
    num_colors = len(color_order)
    colors = [pal(i / (num_colors - 1)) for i in range(num_colors)]
    color_dict = {color_order[i]: colors[i] for i in range(num_colors)}
    return color_dict


def plot_raincloud(df, id_col, data_col_names, data_col_colors, save_path, save_name,
                   x_title, x_name_dict, title,
                   y_title, ymin, ymax, yskip, y_ticks=None, y_jitter=0,
                   data_col_violin_left=None, violin_alpha=0.65, violin_width=0.5, group_spacing=0.5,
                   marker_spread=0.1, marker_size=100, marker_alpha=0.25, scatter_lines=True, format="svg",
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
            violin['cmeans'].get_paths()[0].vertices[:, 0] = np.clip(violin['cmeans'].get_paths()[0].vertices[:, 0], -np.inf, m)
        else:
            violin['cmeans'].get_paths()[0].vertices[:, 0] = np.clip(violin['cmeans'].get_paths()[0].vertices[:, 0], m, np.inf)

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
            plt.scatter(x=id_x_loc, y=id_y_jitter, marker="o", color=data_color, s=marker_size, alpha=marker_alpha, edgecolor=data_color, zorder=2)

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
        plt.yticks(ticks=[y for y in np.arange(ymin, ymax, yskip)], labels=[item for item in y_ticks], fontsize=15)
    plt.xlabel(x_title, fontsize=22, labelpad=10)
    plt.xticks(ticks=[(idx * group_spacing) for idx, item in enumerate(data_col_names)], labels=[x_name_dict[item] for item in data_col_names], fontsize=15)
    plt.title(title, fontsize=25)

    # save plot
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(size_inches_x, size_inches_y)
    if format == "svg":
        plt.savefig(os.path.join(save_path, f"{save_name}.svg"), format="svg", dpi=1000, bbox_inches='tight', pad_inches=0.01)
    if format == "png":
        plt.savefig(os.path.join(save_path, f"{save_name}.png"), format="png", dpi=1000, bbox_inches='tight', pad_inches=0.01)
    del figure
    plt.close()
    return


def plot_pie(categories_names, categories_counts, title, save_path, save_name,
             format="svg", pie_direction=180, categories_colors=None, categories_labels=None,
             annot_groups=True, annot_group_selection=None, annot_props=True,
             legend=False, legend_order=None, legend_vertical=True,
             edge_color="white"):
    if categories_colors is not None:
        categories_colors_list = [categories_colors[cat] for cat in categories_names]
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
                                      colors=categories_colors_list, rotatelabels=True, labeldistance=1.01,
                                      startangle=startangle, textprops={"fontsize": 18},
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

        if legend_order is None:
            ax.legend(wedges, categories_labels_list, loc=location, bbox_to_anchor=anchor, ncol=n_col, fontsize=18)
        else:
            label_to_wedge_color = {label: (wedge, color) for label, wedge, color in
                                    zip(categories_labels_list, wedges, categories_colors_list)}

            ordered_wedges = [label_to_wedge_color[label][0] for label in legend_order if label in label_to_wedge_color]
            ordered_labels = [label for label in legend_order if label in label_to_wedge_color]
            ax.legend(ordered_wedges, ordered_labels, loc=location, bbox_to_anchor=anchor, ncol=n_col, fontsize=17)

    # save
    figure = plt.gcf()  # get current figure
    if format == "svg":
        plt.savefig(os.path.join(save_path, f"{save_name}.svg"), format="svg", dpi=1000, bbox_inches='tight', pad_inches=0.01)
    if format == "png":
        plt.savefig(os.path.join(save_path, f"{save_name}.png"), format="png", dpi=1000, bbox_inches='tight', pad_inches=0.01)
    plt.close()
    return


def plot_pca_scatter_2d(df, hue, title, save_path, pal=None, format="png", annotate=True, size=150):

    if pal is None:
        sns.scatterplot(x='PC1', y='PC2', data=df, hue=hue, s=size,
                        palette=sns.color_palette("hls", len(df[hue].unique().tolist())), legend=False)
    else:
        sns.scatterplot(x='PC1', y='PC2', data=df, hue=hue, palette=pal, s=size, legend=False)
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
    plt.savefig(os.path.join(save_path, f"PCA_result.{format}"), format=f"{format}", dpi=1000, bbox_inches='tight',
                pad_inches=0.01)
    del figure
    plt.close()
    return


def get_labels(label_dict):
    """
    * A HELPER FUNCTION TO plot_binary_preferences *
    Get the labels for "0" and "1" tags in a nested dictionary structure,
    where key=some category name, and value=a dictionary where keys are 0, 1 and values are their labels
    """
    label_0 = [key for key, value in label_dict.items() if value == 0][0]
    label_1 = [key for key, value in label_dict.items() if value == 1][0]
    return label_0, label_1


def plot_binary_preferences(means, sems, colors, labels, label_map, title, save_name, save_path, format="png"):
    fig, ax = plt.subplots(figsize=(16, 6))
    y_pos = np.arange(len(means))

    # ax.barh(y=y_pos, width=means, color=['#1f77b4' if val > 0 else '#ff7f0e' for val in means])  # if I want bars
    # ax.scatter(x=means, y=y_pos, color=['#1f77b4' if val > 0 else '#ff7f0e' for val in means], s=100)  # scatter w/o error bars

    # Plot dots with error bars and differential colors
    for i, (avg, sem, color) in enumerate(zip(means, sems, colors)):
        # plot a dashed line
        ax.hlines(y=y_pos[i], xmin=-1.05, xmax=1.05, color='lightgray', linestyle='--', linewidth=1)
        # plot the actual data
        ax.errorbar(avg, y_pos[i], xerr=sem, fmt='o', color=color, markersize=8,
                    ecolor="black", elinewidth=2, capsize=4)

    ax.axvline(0, color='black', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(["" for i in labels])
    ax.set_xlabel("No Preference", fontsize=15)
    ax.set_title(title, fontsize=15)
    ax.invert_yaxis()
    ax.set_xlim([-1, 1])
    for i, label in enumerate(labels):
        label_0, label_1 = get_labels(label_map[label])
        ax.text(-1.05, i, label_0, va='center', ha='right', fontsize=15, color='black')
        ax.text(1.05, i, label_1, va='center', ha='left', fontsize=15, color='black')

    # save plot
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(15, 12)
    plt.savefig(os.path.join(save_path, f"{save_name}.{format}"), format=f"{format}", dpi=1000, bbox_inches='tight',
                pad_inches=0.01)
    del figure
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
                        s=40,marker='o', c="black")

    for i in range(0, len(x), 3):  # Step by 3 for triplets
        if i + 2 < len(x):  # Ensure we have a full triplet
            ax.plot([x[i], x[i + 1], x[i + 2]],
                    [y[i], y[i + 1], y[i + 2]],
                    [z[i], z[i + 1], z[i + 2]],
                    color="gray")

    if not save:
        plt.show()

    else:
        # save plot
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(15, 12)
        plt.savefig(os.path.join(save_path, f"{save_name}.{save_format}"), format=f"{save_format}",
                    dpi=1000, bbox_inches="tight", pad_inches=0.01)
        del figure
        plt.close()
    return


def plot_pca_scatter(df, hue, title, save_path, pal=None):

    plot_3d_scatter(x_col="PC1", y_col="PC2", z_col="PC3", data=df, c_values_col=hue,
                    color_map=ListedColormap(sns.color_palette("hls", len(df[hue].unique().tolist())).as_hex()),
                    save=False)

    """
    fig = plt.figure(figsize=(10, 7))

    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    sc = ax.scatter(df["PC1"].tolist(), df["PC2"].tolist(), df["PC3"].tolist(),
                    s=40, c=df[hue].tolist(), marker='o',
                    cmap=ListedColormap(sns.color_palette("hls", len(df[hue].unique().tolist())).as_hex()), alpha=1)
    """

    # TODO FINISH THIS
    if pal is None:
        sns.scatterplot(x='PC1', y='PC2', data=df, hue=hue, s=150, palette=sns.color_palette("hls", len(df[hue].unique().tolist())), legend=False)
    else:
        sns.scatterplot(x='PC1', y='PC2', data=df, hue=hue, palette=pal, s=150, legend=False)
    for item in df.index:
        plt.text(df.loc[item, 'PC1'], df.loc[item, 'PC2'], item)
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    # save plot
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(15, 12)
    plt.savefig(os.path.join(save_path, f"PCA_result.svg"), format="svg", dpi=1000, bbox_inches='tight',
                pad_inches=0.01)
    del figure
    plt.close()

    return


def plot_histogram(df, category_col, data_col, save_path, save_name, format="svg"):

    sns.set_style("ticks")
    sns.despine(right=True, top=True)
    plt.rcParams['font.family'] = "Calibri"

    hist_plot = sns.barplot(data=df, x=category_col, y=data_col)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=22)
    plt.xlabel(category_col.title(), fontsize=20)

    # save plot
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(20, 12)
    plt.savefig(os.path.join(save_path, f"{save_name}.{format}"), format=f"{format}", dpi=1000, bbox_inches='tight',
                pad_inches=0.01)
    del figure
    plt.close()

    return


def plot_categorical_bars(categories_prop_df, category_col, data_col, categories_colors,
                          save_path, save_name, format="svg", y_min=0, y_max=100, y_skip=10):
    plt.figure(figsize=(8, 6))
    sns.set_style("ticks")
    plt.rcParams['font.family'] = "Calibri"
    if categories_colors is None:
        barplot = sns.barplot(x=category_col, y=data_col, data=categories_prop_df)
    else:
        barplot = sns.barplot(x=category_col, y=data_col, data=categories_prop_df, palette=categories_colors)

    # add percentages on top of each bar
    for index, row in categories_prop_df.iterrows():
        barplot.text(
            index,  # X-coordinate (position of the bar)
            row[data_col] + 1,  # Y-coordinate (slightly above the bar)
            f"{row[data_col]:.2f}%",  # The text (percentage value)
            color="black",  # Text color
            ha="center",  # Horizontal alignment
            fontsize=25  # Font size
        )

    # now delete y-axis
    sns.despine(right=True, top=True, left=True)
    plt.ylabel("")
    plt.yticks([])


    plt.xticks(fontsize=16)
    plt.xlabel(category_col.title(), fontsize=20)

    # save plot
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(15, 12)
    plt.savefig(os.path.join(save_path, f"{save_name}.{format}"), format=f"{format}", dpi=1000, bbox_inches='tight',
                pad_inches=0.01)
    del figure
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
    plt.close()

    return


def plot_stacked_proportion_bars(plot_data, num_plots, colors, num_ratings, save_path, save_name, title, legend=None,
                                 sem_line=True, ytick_visible=True, text_width=max_text_width):

    fig, axs = plt.subplots(num_plots, 1, figsize=(15, (num_ratings + 1) * num_plots), sharex=True)

    # plot each column as a separate bar
    for i, (col, data) in enumerate(plot_data):
        proportions = data['Proportion']  # between 0-100
        mean_rating = data['Mean']
        std_dev = data['Std Dev']

        # stacked bar plot
        bottom = 0  # left is bottom, as all bars will actually be horizontal

        if num_plots == 1:
            a = axs
        else:
            a = axs[i]

        for j, (response_type, proportion) in enumerate(proportions.items()):
            a.barh(col, proportion, color=colors[j], label=f'Response {int(response_type)}', edgecolor='none',
                   left=bottom)
            bottom += proportion

        # plot the mean rating as a dot
        mean_position = (mean_rating / num_ratings) * 100
        a.plot(mean_position, 0, 'ro', markersize=5, color="#333333")

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
            std_dev_position = (std_dev / num_ratings) * 100
            sem_position = (std_dev / np.sqrt(num_ratings)) * 100
            a.errorbar(mean_position, 0, xerr=sem_position, fmt='o', color="#333333",
                            ecolor="#333333", elinewidth=1, capsize=4, capthick=1, label="")

        # customize each subplot
        a.tick_params(axis='x', labelsize=18)
        if ytick_visible:
            wrapped_col_name = textwrap.fill(col, width=text_width)  # limit the text line width
            a.set_yticklabels([wrapped_col_name], fontsize=18)
        else:
            a.set_yticklabels("")
            a.spines["left"].set_visible(False)
        a.set_xticks([0, 25, 50, 75, 100])
        a.set_xlim(0, 100)  # Set x-axis limits from 0 to 100

    # finalize figure
    fig.text(0.5, 0.06, "Proportion of Responses (%)", ha="center", fontsize=18)
    fig.suptitle(f"{title}", fontsize=25, y=0.94)

    # legend
    if legend is not None:
        handles = [plt.Line2D([0], [0], color=colors[i], lw=10) for i in range(len(legend))]
        fig.legend(
            handles=handles,
            labels=legend,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.925),  # adjust y value to position the legend
            ncol=len(legend),  # arrange in a single row
            frameon=False,
            fontsize=18
        )

    # save plot
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(18, 12)
    plt.savefig(os.path.join(save_path, f"{save_name}.png"), format="png", dpi=1000, bbox_inches='tight',
                pad_inches=0.01)
    del figure
    plt.close()

    return


def plot_scatter_xy(df, identity_col, x_col, x_label, x_min, x_max, x_ticks, y_col, y_label, y_min, y_max, y_ticks,
                    save_path, save_name, annotate_id=True, title_text="", palette_bounds=None, format="png", size=300,
                    corr_line=False, individual_df=None, id_col=None):

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
    if corr_line:
        sns.regplot(data=df, x=x_col, y=y_col, scatter=False, ci=None, order=1,  # linear
                    line_kws=dict(color="black", lw=1.5, linestyle="--"))

    # scatter plot
    sns.scatterplot(data=df, x=x_col, y=y_col, cmap=cmap, norm=norm, c=df["combined"], s=size, zorder=3)

    # annotate
    if annotate_id:
        for i in range(len(df)):
            plt.text(df[x_col][i], df[y_col][i] + 0.065, df[identity_col][i],
                     fontsize=14, ha="center")
    # titles etc
    plt.yticks(np.arange(y_min, y_max + 1 / y_ticks, y_ticks), fontsize=16)
    plt.xticks(np.arange(x_min, x_max + 1 / x_ticks, x_ticks), fontsize=16)
    plt.xlim([x_min, x_max + (1 / x_ticks)])
    plt.ylim([y_min, y_max + (1 / y_ticks)])
    plt.xlabel(x_label.title(), fontsize=18)
    plt.ylabel(y_label.title(), fontsize=18)
    plt.title(f"{title_text.title()}", fontsize=16)

    # save plot
    sns.despine(right=True, top=True)
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(18, 12)
    plt.savefig(os.path.join(save_path, f"{save_name}.{format}"), format=format, dpi=1000, bbox_inches="tight",
                pad_inches=0.01)
    del figure
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
    for category in category_order:
        sns.stripplot(x=category_col, y="jittered", data=df[df[category_col] == category],
                      jitter=horizontal_jitter, size=15, color=category_color_dict[category], alpha=0.8, zorder=1)

    # Plot means with standard deviations manually
    for i, category in enumerate(category_order):
        mean = category_means[category]
        std = category_stds[category]
        sem = category_sems[category]
        plt.plot(i, mean, "o", color="black", markersize=10, label=f"Mean {category}", zorder=2)
        # yerr=sem or yerr=std, whichever we prefer
        plt.errorbar(i, mean, yerr=sem, fmt="o", color="black", capsize=7, zorder=2)

    if y_min is None:
        y_min = df[data_col].min()
    if y_max is None:
        y_max =  df[data_col].max() + 1
    if y_skip is None:
        y_skip = 1
    plt.yticks(np.arange(y_min,y_max + 1 / y_skip, y_skip), fontsize=16)
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
    plt.close()

    return


def plot_world_map_proportion(country_proportions_df, data_column, save_path):

    # load a world map
    world = gpd.read_file(r"C:\Users\Rony\Documents\projects\ethics\survey_analysis\code\ethics_survey\ne_110m_admin_0_countries\ne_110m_admin_0_countries.shp")
    world = world[world["CONTINENT"] != "Antarctica"]  # remove Antarctica
    world = world[world["CONTINENT"] != "Seven seas (open ocean"]  # remove the seven seas
    world.loc[world["NAME"] == "Russia", "CONTINENT"] = "Eastern Europe"  # Russia is otherwise clustered with the whold of Europe

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
    world_with_props = new_world[~new_world["proportion"].isna()]
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

    fig, ax = plt.subplots(1, 1, figsize=(20, 14))
    new_world.boundary.plot(ax=ax, linewidth=1, edgecolor="#050b0c")
    new_world.plot(column="proportion", ax=ax, legend=True,
                   cmap=sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True),
                   missing_kwds={"color": "white"},
                   legend_kwds={'label': "Proportion by Country",
                                'orientation': "horizontal",
                                'shrink': 0.4})

    ax.axis('off')

    # save
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(16, 12)
    plt.savefig(os.path.join(save_path, f"proportion_by_country.png"), format="png", dpi=1000, bbox_inches='tight',
                pad_inches=0.01)
    del figure
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

    fig, ax = plt.subplots(1, 1, figsize=(20, 14))
    new_world.boundary.plot(ax=ax, linewidth=1, edgecolor="#050b0c")
    new_world.plot(column="continent_proportion", ax=ax, legend=True,
                   cmap=sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True),
                   missing_kwds={"color": "white"},
                   legend_kwds={'label': "Proportion by Continent",
                                'orientation': "horizontal",
                                'shrink': 0.4})
    ax.axis('off')

    # save
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(18, 12)
    plt.savefig(os.path.join(save_path, f"proportion_by_continent.png"), format="png", dpi=1000, bbox_inches='tight',
                pad_inches=0.01)
    del figure
    plt.close()

    """
    Save df with data
    """
    new_world_save = new_world.loc[:, ["NAME", "proportion", "CONTINENT", "continent_proportion"]]
    new_world_save.to_csv(os.path.join(save_path, "proportion_geographic.csv"), index=False)

    return


def plot_density(df, x_col, x_col_name, hue_col, hue_col_name, save_name, save_path, format="png", pal="crest", a=0.3, xskip=1):

    ax = sns.kdeplot(data=df, x=x_col, hue=hue_col, fill=True, common_norm=False, palette=pal, alpha=a, linewidth=0)

    # axes
    plt.ylabel("Density", fontsize=17, labelpad=10)
    plt.yticks([y for y in np.arange(0, 1 + (0.1/2), 0.1)], fontsize=15)
    plt.xlabel(x_col_name.title(), fontsize=17, labelpad=10)
    plt.xticks([x for x in np.arange(df[x_col].min(), df[x_col].max() + (xskip/2), xskip)], fontsize=15)

    # legend
    labels = sorted([int(n) for n in df[hue_col].unique().tolist()])
    colors = [plt.Line2D([0], [0], color=sns.color_palette(pal, len(labels))[i], lw=10, alpha=a+0.1) for i in range(len(labels))]
    ax.legend(colors, labels, title=hue_col_name.title(), fontsize=15, title_fontsize=18, ncol=len(labels), frameon=False,
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
    plt.close()
    return



