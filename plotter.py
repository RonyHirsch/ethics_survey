import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Modify the default font settings
mpl.rcParams['font.family'] = 'sans-serif'  # Change to serif fonts
mpl.rcParams['font.sans-serif'] = ['Verdana']  # Specify a serif font


def plot_raincloud(df, id_col, data_col_names, data_col_colors, save_path, save_name,
                   x_title, x_name_dict, title,
                   y_title, ymin, ymax, yskip, y_ticks=None, y_jitter=0,
                   data_col_violin_left=None, violin_alpha=0.65, violin_width=0.5, group_spacing=0.5,
                   marker_spread=0.1, marker_size=100, marker_alpha=0.25, scatter_lines=True, format="svg"):
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
    figure.set_size_inches(15, 12)
    if format == "svg":
        plt.savefig(os.path.join(save_path, f"{save_name}.svg"), format="svg", dpi=1000, bbox_inches='tight', pad_inches=0.01)
    if format == "png":
        plt.savefig(os.path.join(save_path, f"{save_name}.png"), format="png", dpi=1000, bbox_inches='tight', pad_inches=0.01)
    del figure
    plt.close()
    return


def plot_pie(categories_names, categories_counts, categories_colors, categories_labels, title,
             save_path, save_name, format="svg"):
    categories_colors_list = [categories_colors[cat] for cat in categories_names]
    categories_labels_list = [categories_labels[cat] for cat in categories_names]

    # pie order
    startangle = 180 - (categories_counts[0] / sum(categories_counts)) * 180

    # plot
    fig, ax = plt.subplots(figsize=(20, 10))
    wedges, texts, autotexts = ax.pie(categories_counts, labels=categories_labels_list, autopct='%1.0f%%',
                                      colors=categories_colors_list, rotatelabels=True, labeldistance=1.01,
                                      startangle=startangle, textprops={'fontsize': 18})

    # add a title and subtitle
    ax.set_title(title, fontsize=18, fontweight='normal')

    # save
    figure = plt.gcf()  # get current figure
    if format == "svg":
        plt.savefig(os.path.join(save_path, f"{save_name}.svg"), format="svg", dpi=1000, bbox_inches='tight', pad_inches=0.01)
    if format == "png":
        plt.savefig(os.path.join(save_path, f"{save_name}.png"), format="png", dpi=1000, bbox_inches='tight', pad_inches=0.01)
    plt.close()
    return
