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
from scipy.stats import ttest_ind
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


def perform_PCA(df_pivot, save_path, save_name, components=2, clusters=3, label_map=None, binary=True, threshold=0):

    txt_output = list()

    # Perform PCA
    pca = PCA(n_components=components)
    pca_result = pca.fit_transform(df_pivot)

    # Create a DataFrame for PCA results
    """
    PC1 is the direction (in the original feature space) along which the data varies the most. In this case, 
    it represents a pattern where certain creatures consistently receive high or low ratings across both Consciousness
    and Moral Status. 
    """
    pca_df = pd.DataFrame(data=pca_result, columns=["PC1", "PC2"], index=df_pivot.index)
    pca_df.to_csv(os.path.join(save_path, f"{save_name}_PCA_result.csv"), index=True)

    # Get the loadings (coefficients)
    """
    Loadings indicate how much each original variable (ratings for C and MS) contributes to each principal component. 
    They are the coefficients of the linear combinations that make up the principal components.
    If the loadings for PC are both positive and large for both ratings, it suggests that PC1 represents a pattern 
    where creatures receive similar ratings for both C and MS. If it's negative, it's divergence.
    """
    loadings = pd.DataFrame(pca.components_.T, index=df_pivot.columns, columns=["PC1", "PC2"])
    #print(loadings)
    txt_output.append(loadings)

    ### STATISTICAL SIGNIFICANCE

    # fit for statistical significance
    pca.fit(df_pivot)
    """
    The amount of variance each principal component explains how much of the total variance in the data is captured by 
    each component.
    """
    original_variance = pca.explained_variance_ratio_
    # Explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    line = f"Explained Variance Ratio: PC1 = {explained_variance[0]:.2f}, PC2 = {explained_variance[1]:.2f}"
    print(line)
    txt_output.append(line)

    # permutation test: assess the significance of the explained variance of each PC
    n_permutations = 1000
    permuted_variances = np.zeros((n_permutations, 2))

    for i in range(n_permutations):
        # Shuffle each column independently
        permuted_data = shuffle(df_pivot, random_state=i)
        permuted_pca = PCA(n_components=components)
        permuted_pca.fit(permuted_data)
        permuted_variances[i] = permuted_pca.explained_variance_ratio_

    # Calculate p-values: how often the variance explained by the permuted PCs is >= the variance explained by the original PCs
    p_values = np.mean(permuted_variances >= original_variance, axis=0)
    line = f"P-values for explained variance: PC1 = {p_values[0]:.4f}, PC2 = {p_values[1]:.4f}"
    print(line)
    txt_output.append(line)

    ### PLOT
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=clusters, random_state=42, n_init=10)  # The n_init parameter controls the number of times the KMeans algorithm is run with different centroid seeds; 10 is the default
    df_pivot["Cluster"] = kmeans.fit_predict(df_pivot)
    unified_df = pd.merge(df_pivot, pca_df, left_index=True, right_index=True)
    unified_df.to_csv(os.path.join(save_path, f"{save_name}_pca_result_with_kmeans_clusters.csv"), index=True)  # index here is the identity of each dot
    plotter.plot_pca_scatter_2d(df=unified_df, hue=unified_df["Cluster"], title="Ratings PCA",
                                save_path=save_path, save_name=save_name,
                                pal=["#A3333D", "#27474E"], annotate=False, size=250)  # "#B1740F", "#003554"

    # test the statistical significance of the clustering
    """
    The silhouette score measures how similar a data point is to its own cluster compared to other clusters. 
    It ranges from -1 to 1, where a value closer to 1 indicates that the data points are well clustered.
    """

    # calculate the silouhette score of the real data
    silhouette_avg = silhouette_score(df_pivot.drop(columns="Cluster"), df_pivot["Cluster"])
    line = f"{clusters}-Means Clustering Silhouette Score: {silhouette_avg:.2f}"
    print(line)
    txt_output.append(line)

    # random permutations
    n_iterations = 1000
    random_silhouette_scores = list()
    for _ in range(n_iterations):
        random_clusters = np.random.randint(0, clusters, size=df_pivot.shape[0])
        score = silhouette_score(df_pivot.drop(columns="Cluster"), random_clusters)
        random_silhouette_scores.append(score)
    p_value = np.mean(silhouette_avg < np.array(random_silhouette_scores))
    line = f"P-value of Silhouette Score compared to random clusters ({n_iterations} iterations): {p_value:.4f}"
    print(line)
    txt_output.append(line)


    """
    Identify the centroids: the choices that each cluster made on average on each question
    """
    cluster_centroids = unified_df.groupby("Cluster").mean()
    cluster_centroids.to_csv(os.path.join(save_path, f"{save_name}_cluster_centroids_raw.csv"), index=True)  # index=Cluster id

    # plot the centroids
    cluster_centroids_sem = unified_df.groupby("Cluster").sem()  # the centroids were mean, these are sems for plotting

    # for each cluster
    for cluster in range(0, clusters):
        # take cluster means
        cluster_centroid = cluster_centroids.iloc[[cluster], :]  # take only 1 row (the row of the cluster)
        cluster_centroid = cluster_centroid.drop(columns=["PC1", "PC2"])  # drop the non-question columns
        # take cluster sems
        cluster_sems = cluster_centroids_sem.iloc[[cluster], :]
        cluster_sems = cluster_sems.drop(columns=["PC1", "PC2"])

        if binary:
            """
            map binary choices to a -1 to 1 scale where 0.5 means no preference, such that 
            -1 indicates complete preference for "0"
            +1 indicates complete preference for "1"
            0 means no preference (50-50)
            """
            preferences = cluster_centroid.iloc[0] * 2 - 1  # scale the means
            labels = preferences.index
            """
            Unlike the means, SEMs are measures of spread (uncertainty), not actual preference values. 
            So, they should only be scaled (by the same factor as the means), not shifted (no -1), as it simply 
            represents how much the mean might fluctuate, not a direct preference value. 
            We only need to stretch its range, not change its center
            """
            sems_scaled = cluster_sems.iloc[0] * 2

            # Define colors based on the preferences: positive for "1", negative for "0"
            colors = ["#102E4A" if val > 0 else "#EDAE49" for val in preferences]

            plotter.plot_binary_preferences(means=preferences, sems=sems_scaled, colors=colors,
                                            labels=labels, label_map=label_map, title=f"Cluster {cluster}",
                                            save_name=f"{save_name}_cluster_{cluster}_centroids", save_path=save_path)

        else:
            """
            choices are not binary - so they were not 0/1 which can be scaled to -1/1. 
            keep them as they are in the same scale
            """
            preferences = cluster_centroid.iloc[0]  # Keep the means as they are (do not scale)
            labels = preferences.index
            sems_scaled = cluster_sems.iloc[0]  # no need to change it either
            colors = ["#DB5461" if val <= threshold else "#26818B" for val in preferences]
            plotter.plot_nonbinary_preferences(means=preferences, sems=sems_scaled, colors=colors, min=1, max=4,
                                               thresh=threshold, labels=labels, label_map=label_map,
                                               title=f"Cluster {cluster}",
                                               save_name=f"{save_name}_cluster_{cluster}_centroids", save_path=save_path)

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

    # WRITE TO TXT
    with open(os.path.join(save_path, f"{save_name}_PCA_result.txt"), "w") as file:
        for line in txt_output:
            line_str = str(line)
            file.write(line_str + '\n')

    return unified_df


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


def compute_stats(column):
    proportions = column.value_counts(normalize=True).sort_index() * 100
    mean_rating = column.mean()
    std_dev = column.std()
    return proportions, mean_rating, std_dev


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



