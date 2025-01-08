import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from kmodes.kprototypes import KPrototypes
import scipy.stats as stats
from scipy.stats import chi2_contingency
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind, ttest_1samp
from scipy.spatial.distance import cdist
from sklearn.utils import shuffle
import plotter


def chi_squared_test(contingency_table):
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    result_df = pd.DataFrame({
        "test": ["chi squared"],
        "statistic": [chi2],
        "p": [p],
        "df": [dof],  # Degrees of Freedom
        #"expected": [expected]  # Expected Frequencies
    })
    return result_df


def independent_samples_ttest(list_group1, list_group2):  # continuous data
    t_stat, p_value = ttest_ind(list_group1, list_group2)
    result_df = pd.DataFrame({
        "test": ["independent t-test"],
        "statistic": [t_stat],
        "p": [p_value],
        "df": [len(list_group1) + len(list_group2) - 2]
    })
    return result_df


def one_sample_ttest(list_group1, test_value=0, ci=0.95):
    ttest_result = ttest_1samp(list_group1, test_value)
    conf_interval = ttest_result.confidence_interval(confidence_level=ci)
    result_df = pd.DataFrame({
        "test": ["one sample t-test"],
        "statistic": [ttest_result.statistic],
        "p": [ttest_result.pvalue],
        "df": [ttest_result.df],
        f"{ci} CI low": [conf_interval.low],
        f"{ci} CI high": [conf_interval.high]
    })
    return result_df


def mann_whitney_utest(list_group1, list_group2):  # ordinal data
    u_stat, p_value = mannwhitneyu(list_group1, list_group2, alternative='two-sided')
    result_df = pd.DataFrame({
        "test": ["mann whitney"],
        "statistic": [u_stat],
        "p": [p_value],
        "df": [len(list_group1) + len(list_group2) - 2]
    })

    return result_df


def lca_analysis(df, save_path, n_classes=3):
    """
    Perform LCA analysis on df, assuming each row = sub and each col = some ordinal rating of something.
    """

    # save df for LCA in R
    #df.to_csv(os.path.join(save_path, "LCA_df.csv"), index=False)

    """
    Python doesn't have direct support in LCA. So we will save the scaled df to be analyzed in R.
    In the meantime, we will use Gaussian Mixture Model, to approximate the probability of latent classes. 
    This model can estimate the probabilities of belonging to latent classes.
    """

    # assuming the ratings are ordinal, standardize them before LCA
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    # fit the Gaussian Mixture Model as an approximation to LCA
    gmm = GaussianMixture(n_components=n_classes, random_state=42)
    gmm.fit(df_scaled)
    latent_classes = gmm.predict(df_scaled)  # get the predicted class (latent class) for each participant
    class_probs = gmm.predict_proba(df_scaled)
    df["latent_class"] = latent_classes

    df.to_csv(os.path.join(save_path, "LCA_df_GausMM.csv"), index=False)

    return


def perform_PCA(df_pivot, save_path, save_name, components=2):
    """
    PCA reduces the dimensionality of the given df_pivot. We compute and saves the PCA-transformed data (PC1, PC2) and
    the loadings (coefficients showing how much each feature contributes to the principal components). We then assess
    the explained variance of each principal component and test its statistical significance using permutation tests.
    :return: pca_df - the PCA-transformed data; the explained variance ratios; the loadings (describing the relationship
    between the original features and principal components)
    """
    txt_output = list()

    # Perform PCA
    pca = PCA(n_components=components)
    pca_result = pca.fit_transform(df_pivot)

    # Create a DataFrame for PCA results
    """
    PC1 is the direction (in the original feature space) along which the data varies the most. 
    For example, in rating C and MS, it represents a pattern where certain creatures consistently receive high or low 
    ratings across both Consciousness and Moral Status. 
    """
    pca_df = pd.DataFrame(data=pca_result, columns=[f"PC{i+1}" for i in range(components)], index=df_pivot.index)
    pca_df.to_csv(os.path.join(save_path, f"{save_name}_PCA_result.csv"), index=True)

    # Get the loadings (coefficients)
    """
    Loadings indicate how much each original variable (e.g., the ratings of C and MS) contributes to each principal 
    component. They are the coefficients of the linear combinations that make up the principal components.
    If the loadings for PC are both positive and large for both ratings, it suggests that PC1 represents a pattern 
    where creatures receive similar ratings for both C and MS. If it's negative, it's divergence.
    """
    loadings = pd.DataFrame(pca.components_.T, index=df_pivot.columns, columns=[f"PC{i+1}" for i in range(components)])
    txt_output.append(loadings)

    # Explained variance ratio
    """
    The amount of variance each principal component explains how much of the total variance in the data is captured by 
    each component.
    """
    explained_variance = pca.explained_variance_ratio_
    line = f"Explained Variance Ratio: " + ", ".join([f"PC{i+1} = {var:.2f}" for i, var in enumerate(explained_variance)])
    print(line)
    txt_output.append(line)

    # Permutation test
    n_permutations = 1000
    permuted_variances = np.zeros((n_permutations, components))

    for i in range(n_permutations):
        permuted_data = shuffle(df_pivot, random_state=i)
        permuted_pca = PCA(n_components=components)
        permuted_pca.fit(permuted_data)
        permuted_variances[i] = permuted_pca.explained_variance_ratio_

    # Calculate p-values: how often the variance explained by the permuted PCs is >= the variance explained by the original PCs
    p_values = np.mean(permuted_variances >= explained_variance, axis=0)
    line = f"P-values for explained variance: " + ", ".join([f"PC{i+1} = {p:.4f}" for i, p in enumerate(p_values)])
    print(line)
    txt_output.append(line)

    with open(os.path.join(save_path, f"{save_name}_PCA_result.txt"), "w") as file:
        for line in txt_output:
            file.write(str(line) + '\n')

    return pca_df, loadings, explained_variance


def perform_kmeans(df_pivot, save_path, save_name, clusters=2, normalize=False):
    """
    Perform k-means clustering (scikit-learn's) to group the data into a specified number of clusters.
    We then append the cluster labels to the dataset and calculate the silhouette score,
    which evaluates how well-separated the clusters are. We then test the clustering's statistical significance by
    comparing it to random clusters. This medho also saves the cluster centroids, which represent the average values
    of features for each cluster.
    """
    txt_output = list()

    # if normalize is True, normalize the data
    if normalize:
        scaler = StandardScaler()
        df_pivot = scaler.fit_transform(df_pivot)

    # Perform k-means clustering
    # The n_init parameter controls the number of times the KMeans algorithm is run with different centroid seeds; 10 is the default
    kmeans = KMeans(n_clusters=clusters, random_state=42, n_init=10)
    df_pivot["Cluster"] = kmeans.fit_predict(df_pivot)

    # Calculate silhouette score
    """
    The silhouette score measures how similar a data point is to its own cluster compared to other clusters. 
    It ranges from -1 to 1, where a value closer to 1 indicates that the data points are well clustered.
    """
    silhouette_avg = silhouette_score(df_pivot.drop(columns="Cluster"), df_pivot["Cluster"])
    line = f"{clusters}-Means Clustering Silhouette Score: {silhouette_avg:.2f}"
    print(line)
    txt_output.append(line)

    # test the statistical significance of the clustering (against random clusters)
    n_iterations = 1000
    random_silhouette_scores = [
        silhouette_score(df_pivot.drop(columns="Cluster"), np.random.randint(0, clusters, size=df_pivot.shape[0]))
        for _ in range(n_iterations)
    ]
    p_value = np.mean(silhouette_avg < np.array(random_silhouette_scores))
    line = f"P-value of Silhouette Score compared to random clusters ({n_iterations} iterations): {p_value:.4f}"
    print(line)
    txt_output.append(line)

    # Save cluster centroids
    cluster_centroids = df_pivot.groupby("Cluster").mean()
    cluster_centroids.to_csv(os.path.join(save_path, f"{save_name}_cluster_centroids_raw.csv"), index=True)

    """
    Test the significance of the difference between the centroids: 
    test whether there’s a significant association between cluster membership and the choices made on each question. 
    To do that, we'll use a Chi-square test, as the choice is binary and the sample is large. 
    """

    q_cols = df_pivot.columns[:-1]  # everything but the "Cluster" column
    result = list()
    for choice in q_cols:
        # create a contingency table for the current choice and the cluster column
        contingency_table = pd.crosstab(df_pivot["Cluster"], df_pivot[choice])
        # perform the Chi-Square test
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        # Expected: the expected frequencies for each cell in the contingency table, the theoretical frequencies
        # that would occur in each cell of a contingency table if the choices are independent of the cluster
        result.append({"Choice": choice, "Chi2": chi2, "p-value": p, "dof": dof, "Expected": expected})
    chisq_df = pd.DataFrame(result)
    chisq_df.to_csv(os.path.join(save_path, f"{save_name}_cluster_centroids_chisq.csv"), index=False)


    with open(os.path.join(save_path, f"{save_name}_kmeans_{clusters}_result.txt"), "w") as file:
        for line in txt_output:
            file.write(str(line) + '\n')

    return df_pivot, kmeans


def plot_kmeans_on_PCA(df_pivot, pca_df, save_path, save_name, palette=None):
    """
    Gets:
    - "pca_df" from the method perform_PCA, and
    - "df_pivot" from the method perform_kmeans,
    and draws a scatterplot showing the cluster assignments (Kmeans) projected onto the PCA-transformed space (PCA),
    so that we'd have an easier time interpreting the clustering.
    NOTE: we do NOT assume that the PCA was used as input for the kmeans, as "perform_kmeans" calculates clusters on the
    RAW data (not the dim-reduced one).
    """
    unified_df = pd.merge(df_pivot, pca_df, left_index=True, right_index=True)
    unified_df.to_csv(os.path.join(save_path, f"{save_name}_pca_result_with_kmeans_clusters.csv"), index=True)

    # Default palette
    if palette is None:
        palette = ["#A3333D", "#27474E"]

    # Plot PCA scatter with cluster labels
    plotter.plot_pca_scatter_2d(
        df=unified_df,
        hue="Cluster",
        title="Ratings PCA",
        save_path=save_path,
        save_name=save_name,
        pal=palette,
        annotate=False,
        size=250,
        fmt="svg",
    )
    return unified_df


def plot_cluster_centroids(cluster_centroids, cluster_sems, save_path, save_name, label_map=None, binary=True,
                           threshold=0, overlaid=False, cluster_colors_overlaid=None, fmt="png"):
    """
    Plots the centroids for each cluster with preferences and uncertainty.
    Can either plot individual plots per cluster or a single overlaid plot for all clusters.
    """
    if overlaid:
        all_preferences = []
        all_sems = []
        all_colors = []
        cluster_names = []

        if not cluster_colors_overlaid:
            cluster_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Default color palette for clusters
        else:
            cluster_colors = cluster_colors_overlaid

        # Collect data for all clusters
        for cluster in range(len(cluster_centroids)):
            cluster_centroid = cluster_centroids.iloc[[cluster], :]
            cluster_sem = cluster_sems.iloc[[cluster], :]

            if binary:
                # Scale binary preferences
                preferences = cluster_centroid.iloc[0] * 2 - 1
                sems_scaled = cluster_sem.iloc[0] * 2
                # Define colors based on clusters
                colors = [cluster_colors[cluster]] * len(preferences)
            else:
                # Non-binary preferences
                preferences = cluster_centroid.iloc[0]
                sems_scaled = cluster_sem.iloc[0]
                # Define colors based on clusters
                colors = [cluster_colors[cluster]] * len(preferences)

            all_preferences.append(preferences)
            all_sems.append(sems_scaled)
            all_colors.append(colors)
            cluster_names.append(f"Cluster {cluster}")

        # Call the overlaid plot function
        plotter.plot_overlaid_preferences(
            all_preferences=all_preferences,
            all_sems=all_sems,
            all_colors=all_colors,
            labels=preferences.index,
            label_map=label_map,
            cluster_names=cluster_names,
            binary=binary,
            save_name=f"{save_name}_overlaid_centroids",
            save_path=save_path,
            threshold=threshold,
            fmt=fmt
        )
    else:
        # Plot per cluster
        for cluster in range(len(cluster_centroids)):
            cluster_centroid = cluster_centroids.iloc[[cluster], :]
            cluster_sem = cluster_sems.iloc[[cluster], :]

            if binary:
                preferences = cluster_centroid.iloc[0] * 2 - 1
                sems_scaled = cluster_sem.iloc[0] * 2
                labels = preferences.index
                colors = ["#102E4A" if val > 0 else "#EDAE49" for val in preferences]

                plotter.plot_binary_preferences(
                    means=preferences,
                    sems=sems_scaled,
                    colors=colors,
                    labels=labels,
                    label_map=label_map,
                    title=f"Cluster {cluster}",
                    save_name=f"{save_name}_cluster_{cluster}_centroids",
                    save_path=save_path
                )
            else:
                preferences = cluster_centroid.iloc[0]
                sems_scaled = cluster_sem.iloc[0]
                labels = preferences.index
                colors = ["#DB5461" if val <= threshold else "#26818B" for val in preferences]

                plotter.plot_nonbinary_preferences(
                    means=preferences,
                    sems=sems_scaled,
                    colors=colors,
                    min=1,
                    max=4,
                    thresh=threshold,
                    labels=labels,
                    label_map=label_map,
                    title=f"Cluster {cluster}",
                    save_name=f"{save_name}_cluster_{cluster}_centroids",
                    save_path=save_path
                )
    return


def corr_per_item(df, items, save_path):

    item_correlations = {}
    for item in items:
        df_item = df[["response_id", f"c_{item}", f"ms_{item}"]].dropna()
        # correlation between ratings of consciousness and moral status
        p_corr, p_p_value = stats.pearsonr(df_item[f"c_{item}"], df_item[f"ms_{item}"])
        #r_corr, r_p_value = stats.spearmanr(df_item[f"c_{item}"], df_item[f"ms_{item}"])
        item_correlations[f"{item}"] = {'pearson_correlation': p_corr, 'pearson_p_value': p_p_value}
                                        #, 'spearman_correlation': r_corr, 'spearman_p_value': r_p_value}

    correlation_df = pd.DataFrame.from_dict(item_correlations, orient='index').reset_index(drop=False).sort_values(by='pearson_correlation', ascending=False)
    correlation_df.to_csv(os.path.join(save_path, "correlation.csv"), index=False)
    df.to_csv(os.path.join(save_path, "correlation_data.csv"), index=False)
    return


def compute_stats(column, possible_values=None, normalize=True, stats=True):
    if possible_values is None:  # Infer possible values from the column
        possible_values = sorted(column.dropna().unique())

    # Initialize proportions for all possible values
    proportions = {val: 0 for val in possible_values}
    if normalize:
        actual_proportions = column.value_counts(normalize=True).sort_index() * 100
    else:
        actual_proportions = column.value_counts().sort_index()
        actual_proportions = actual_proportions.transform(lambda x: x * 100 / x.sum())
    for value, proportion in actual_proportions.items():
        proportions[value] = proportion
    if stats:
        mean_rating = column.mean()
        std_dev = column.std()
    else:
        mean_rating = None
        std_dev = None
    n = column.count()
    return proportions, mean_rating, std_dev, n


animal_other_conversion = {"Rats": "Rodents",
                           "rat": "Rodents",
                           "Bears": "Bears",
                           "Snails": "Snails",
                           "Hamsters": "Rodents",
                           "Hamster": "Rodents",
                           "Turtle.": "Reptiles",  # turtles are reptiles
                           "birds": "Birds",
                           "rabbits": "Rabbits",
                           "Bunny": "Rabbits",  # bunny is a young rabbit
                           "Frogs": "Frogs",  # amphibians, but we didn't have that
                           "Horses": "Livestock",
                           "goats": "Livestock",
                           "Equines": "Livestock",
                           "Chicken": "Livestock",
                           "invertebrates": "Insects"
                           }


def replace_animal_other(row):
    A = "Please specify which animals"
    B = "animalsExp_Other: please specify"

    if pd.notna(row[A]) and "Other" in row[A] and isinstance(row[B], str):  # if A and B are not NaN
        substrings = row[B].split(',')  # split the free other-string into a list
        replacements = [animal_other_conversion.get(x.strip(), x.strip()) for x in substrings]
        if any(x.strip() in animal_other_conversion for x in substrings):
            replacements_str = ','.join(replacements)

            # replace "Other" in A only if it is a standalone word
            a_list = row[A].split(',')
            a_list = [replacements_str if x.strip() == "Other" else x for x in a_list]
            row[A] = ','.join(a_list)

            # remove converted substrings from B so I'll know it worked
            converted_substrings = [x.strip() for x in substrings if x.strip() in animal_other_conversion]
            remaining_substrings = [x for x in substrings if x.strip() not in converted_substrings]
            row[B] = ','.join(remaining_substrings)
        else:
            row[A] = row[A]  # if no conversion, keep "Other" unchanged
    return row


def calculate_distances(df, x_col, y_col, metric='euclidean'):
    """
    Calculate distances from the diagonal for points in a DataFrame.
    :param df: pd df with the points as x-col and y-col; each row is a point
    :param x_col:
    :param y_col:
    :param metric:str, the distance metric to use (default is 'euclidean'). See scipy.spatial.distance.cdist for available metrics.
    :return: ist of distances for each point in the df
    """
    distances = []
    for _, row in df.iterrows():
        point = np.array([[row[x_col], row[y_col]]])
        diagonal_point = np.array([[row[x_col], row[x_col]]])
        distance = cdist(point, diagonal_point, metric=metric)[0, 0]
        distances.append(distance)
    return distances



