"""
Statistical and machine learning helper functions for survey analysis.

Provides reusable utilities including: random forest pipelines, ordinal regression, mixed-effects models, PERMANOVA,
chi-square tests, PCA, k-means clustering with silhouette-based optimal k selection

Author: RonyHirsch
"""

import os
os.environ["MPLBACKEND"] = "Agg"   # headless backend, no Tkinter
import re
import matplotlib
matplotlib.use("Agg", force=True)  # belt-and-suspenders: force Agg to eliminate Tk as it causes redundant runtime errors - BEFORE shap
import shap
import math
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from scipy.stats import chi2_contingency, mannwhitneyu
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from statsmodels.stats.multitest import multipletests
import plotter


# Custom transformer to merge rare categories
class RareCategoryMerger(BaseEstimator, TransformerMixin):
    def __init__(self, threshold):
        self.threshold = threshold
        self.mappings_ = {}
        self.dropped_categories_ = {}  # to track dropped ones

    def fit(self, X, y=None):
        for col in X.columns:
            X[col] = X[col].astype(str)
            value_counts = X[col].value_counts()
            kept = value_counts[value_counts >= self.threshold].index.tolist()
            dropped = value_counts[value_counts < self.threshold].index.tolist()
            self.mappings_[col] = kept
            self.dropped_categories_[col] = dropped
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in X.columns:
            X_transformed[col] = X_transformed[col].astype(str)
            X_transformed[col] = X_transformed[col].where(X_transformed[col].isin(self.mappings_[col]), 'Other')
        return X_transformed


def run_random_forest_pipeline(dataframe, dep_col, categorical_cols, order_cols, save_path, save_prefix="",
                               rare_class_threshold=5, n_permutations=1000, scoring_method="accuracy",
                               cv_folds=10, split_test_size=0.3, n_repeats=50,
                               shap_plot=True, shap_plot_colors=None, feature="Experience"):

    print("starting RF pipeline")
    # dep_col is assumed to be binary
    df = dataframe.copy()
    df_model = df[categorical_cols + order_cols + [dep_col]].dropna(subset=[dep_col])

    # merge rare categories
    for col in categorical_cols:
        df_model[col] = df_model[col].astype(str)
    print("merging rare categories")
    rare_merger = RareCategoryMerger(threshold=rare_class_threshold)
    df_model[categorical_cols] = rare_merger.fit_transform(df_model[categorical_cols])

    """
    Balance the dataset >>
    we'll us class weighting: RandomForestClassifier's class_weight='balanced' below
    """

    X = df_model[categorical_cols + order_cols]
    y = df_model[dep_col]

    # train/test split (before any modeling)
    print("splitting data into train and test")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=split_test_size)

    # setting up preprocessing pipelines
    print("setting up preprocessing and model pipeline")
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("cat", categorical_transformer, categorical_cols),
        ("num", numerical_transformer, order_cols)
    ])

    # full pipeline
    rf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(class_weight='balanced'))  # "balanced" = to deal with having a majority/minority class
    ])

    # grid search on training data for hyperparameter tuning
    print("grid search for hyperparameter tuning")
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_features': ['sqrt', 'log2']
    }
    # the best model is selected using cross validation using the training set alone
    grid_search = GridSearchCV(rf_pipeline, param_grid, cv=cv_folds, scoring=scoring_method, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"Grid search complete. Best hyperparameters: {grid_search.best_params_}")

    # cross-validate best model on training set
    print("cross-validating best model on training data")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True)
    train_cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring=scoring_method)
    mean_train_cv_score = np.mean(train_cv_scores)
    print(f"Mean train CV {scoring_method}: {mean_train_cv_score:.4f}")

    # refit the best model on full training set
    best_model.fit(X_train, y_train)

    # evaluation on the test set
    print("evaluating on test set")
    test_preds = best_model.predict(X_test)
    if scoring_method == "accuracy":
        test_score = accuracy_score(y_test, test_preds)
    elif scoring_method == "f1":
        test_score = f1_score(y_test, test_preds)
    print(f"Test {scoring_method}: {test_score:.4f}")

    # permutation test on test set
    print(f"running permutation test with {n_permutations} iterations")
    perm_scores = []
    for i in tqdm(range(n_permutations), desc="Permutation test"):
        y_test_permuted = shuffle(y_test, random_state=i).reset_index(drop=True)
        preds = best_model.predict(X_test)
        if scoring_method == "accuracy":
            perm_scores.append(accuracy_score(y_test_permuted, preds))
        elif scoring_method == "f1":
            perm_scores.append(f1_score(y_test_permuted, preds))
    p_value = np.mean([s >= test_score for s in perm_scores])
    print(f"Permutation test p-value: {p_value:.4f}")

    """
    Feature importances - two types:
    (1) Gini importance: used internally by scikit-learn's RF when calling "model.feature_importances_"
     has no direction, and not based on test data. 
    (2) permutation_importance: checks impact of features on prediction performance (reflects actual test performance)
    
    Then, we will also use SHAP to understand the directionality of each feature.  
    SHAP (SHapley Additive exPlanations) analysis is a method for interpreting the predictions of machine learning 
    models by explaining how each feature contributes to the model's output.
    https://shap.readthedocs.io/en/latest/
    """
    print("extracting feature importances")
    X_test_preprocessed = best_model.named_steps['preprocessor'].transform(X_test)
    feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
    model = best_model.named_steps['classifier']

    # gini importances
    gini_importances = model.feature_importances_

    # permutation importances
    if hasattr(X_test_preprocessed, "toarray"):  # create a dense version for permutation_importance ONLY
        X_test_preprocessed_dense = X_test_preprocessed.toarray()
    else:
        X_test_preprocessed_dense = X_test_preprocessed

    perm_result = permutation_importance(
        model,
        X_test_preprocessed_dense,
        y_test,
        n_repeats=n_repeats,  # permutation importance stability, the standard is 30; Empirical studies (including from scikit-learn authors) show that 30–50 repeats  is usually enough to estimate permutation importance reliably and with low variance.
        scoring=scoring_method
    )

    # SHAP values with directionality
    explainer = shap.TreeExplainer(model, model_output="raw")
    if hasattr(X_test_preprocessed, "toarray"):
        X_shap = X_test_preprocessed.toarray().astype(float)
    else:
        X_shap = X_test_preprocessed.astype(float)

    shap_values = explainer.shap_values(X_shap)
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # shape: (n_samples, n_features, n_classes)
        shap_vals = shap_values[:, :, 1]  # class 1
    elif isinstance(shap_values, list) and len(shap_values) == 2:
        # older style: list of arrays for each class
        shap_vals = shap_values[1]
    else:
        raise ValueError(f"Unrecognized SHAP output shape: {type(shap_values)} {getattr(shap_values, 'shape', None)}")
    mean_shap_values = shap_vals.mean(axis=0)  # how much, on average, this feature pushes predictions toward or away from class 1 (positive: increase likelihood of 1)

    # importances df with SHAP values
    feat_idx = pd.Index(feature_names, name="Feature")
    importances_df = pd.DataFrame({"Gini_Importance": gini_importances,
                                   "Perm_Importance_Mean": perm_result.importances_mean,
                                   "Perm_Importance_Std": perm_result.importances_std,
                                   "SHAP_Mean": mean_shap_values,
                                   }, index=feat_idx)

    # add directionality stats for each feature
    X_shap_df = pd.DataFrame(X_shap, columns=feature_names)
    mean_shap_high_list = []
    mean_shap_low_list = []
    direction_list = []

    X_shap_df = pd.DataFrame(X_shap, columns=feature_names)
    for i, fname in enumerate(feature_names):
        """
        In vanilla SHAP analysis, we don’t usually bin features by median and then assign a “direction”, people
        just do it from the shap plot. but I want to summarize the trend. I risk missing U-shaped relationships, 
        but manual inspection of the SHAP plot can take care of that. 
        When I did, she shap features looked monotonic (higher values pushed towards one cluster or the other), 
        so Direction_Toward isn't hiding any big nonlinear effects.
        
        For each feature, we split the test set into two groups:
        High values =  feature values above that feature’s median
        Low values = feature values below or equal to the median
        Then: 
        Mean_SHAP_High = average SHAP value for the high-value group
        Mean_SHAP_Low = average SHAP value for the low-value group
        Finally:
        If Mean_SHAP_High > Mean_SHAP_Low: Direction_Toward = Cluster 1
        (higher values tend to push predictions toward the positive class / "cluster 1")
        Otherwise: Direction_Toward = Cluster 0
        (higher values tend to push predictions toward the negative class / "cluster 0")
        """

        shap_feature_values = shap_vals[:, i]
        median_val = X_shap_df[fname].median()
        high_mask = X_shap_df[fname] > median_val
        low_mask = ~high_mask

        mean_high = shap_feature_values[high_mask].mean()
        mean_low = shap_feature_values[low_mask].mean()

        mean_shap_high_list.append(mean_high)
        mean_shap_low_list.append(mean_low)

        direction_list.append("Cluster 1" if mean_high > mean_low else "Cluster 0")


    direction_df = pd.DataFrame({
        "Mean_SHAP_High": mean_shap_high_list,
        "Mean_SHAP_Low": mean_shap_low_list,
        "Direction_Toward": direction_list,   # which cluster the feature tends to push predictions toward when its value is ABOVE the median
    }, index=feat_idx)

    # add to importances_df
    importances_df = importances_df.join(direction_df)
    # and sort by importance
    importances_df = (importances_df
                      .sort_values("Gini_Importance", ascending=False)
                      .reset_index())

    # save outputs
    df_model.to_csv(os.path.join(save_path, f"{save_prefix}random_forest_df_model.csv"), index=False)
    importances_df.to_csv(os.path.join(save_path, f"{save_prefix}random_forest_feature_importances.csv"), index=False)

    summary_data = {
        #'best_model': best_model,
        'best_params': str(grid_search.best_params_),
        f"test_{scoring_method}_score": test_score,
        f"mean_train_cv_{scoring_method}": mean_train_cv_score,
        'train_cv_scores': ', '.join([f"{s:.3f}" for s in train_cv_scores]),
        'p_value': p_value,
        'dropped_categories': str(rare_merger.dropped_categories_)
    }
    summary_df = pd.DataFrame(list(summary_data.items()), columns=["Metric", "Value"])
    summary_df.to_csv(os.path.join(save_path, f"{save_prefix}random_forest_summary.csv"), index=False)

    if shap_plot:
        """
        SHAP summary plot
        Each dot = one observation
        X-axis = SHAP value: How much that feature pushed the prediction toward one class or the other.
        Negative SHAP = push toward class 0
        Positive SHAP = push toward class 1
        Color = feature value (red = high, blue = low)
        Y-axis = features ranked by importance (from top to bottom)
        """
        import matplotlib.pyplot as plt
        print("generating SHAP summary plot")

        if shap_plot_colors is not None:
            import matplotlib.colors as mcolors
            custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", shap_plot_colors)

        else:
            from matplotlib import cm
            custom_cmap = cm.get_cmap("coolwarm")

        plt.figure(figsize=(10, 6))
        # clean feature names
        prefixes_to_remove = ("num_", "cat_", "onehot__", "scaler__")
        clean_feature_names = [  # order preserved because list comprehension iterates over feature_names in place
            re.sub(rf"^({'|'.join(prefixes_to_remove)})+", "", s) for s in feature_names
        ]
        shap.summary_plot(
            shap_vals,
            X_shap,  # ndarray
            feature_names=clean_feature_names,
            cmap=custom_cmap,  # the custom cmap from above, not colors directly
            plot_type="dot",  # or BAR
            show=False
        )
        plt.xlabel("Impact on probability to belong to class 1", fontsize=20)
        plt.ylabel(f"{feature}", fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{save_prefix}random_forest_shap_summary_plot.svg"),
                    format="svg", dpi=1000)
        plt.close()

    return best_model


def run_group_mann_whitney(df, comparison_cols, group_col, group_col_name, group1_val, group1_name,
                           group2_val, group2_name, p_corr_method="fdr_bh"):
    """
    Run Mann Whitney U test on multiple columns that we want to compare between two groups.
    group_col is the column denoting the two groups,
    comparison_cols are the columns to be compared (one by one with a MW-U test)
    """
    results = []
    for col in comparison_cols:
        group1 = pd.to_numeric(df[df[group_col] == group1_val][col].dropna())
        group2 = pd.to_numeric(df[df[group_col] == group2_val][col].dropna())
        result = mann_whitney_utest(list_group1=group1, list_group2=group2)

        result["Item"] = col

        result[f"N_{group_col_name}={group1_name}"] = len(group1)
        result[f"Mean_{group_col_name}={group1_name}"] = group1.mean()
        result[f"SD_{group_col_name}={group1_name}"] = group1.std(ddof=1)
        result[f"SE_{group_col_name}={group1_name}"] = group1.std(ddof=1) / np.sqrt(len(group1))

        result[f"N_{group_col_name}={group2_name}"] = len(group2)
        result[f"Mean_{group_col_name}={group2_name}"] = group2.mean()
        result[f"SD_{group_col_name}={group2_name}"] = group2.std(ddof=1)
        result[f"SE_{group_col_name}={group2_name}"] = group2.std(ddof=1) / np.sqrt(len(group2))

        results.append(result)

    result_df = pd.concat(results)
    # Correct for multiple comparisons
    raw_pvals = result_df["p"]
    _, corrected_pvals, _, _ = multipletests(raw_pvals, method=p_corr_method)  # benjamini-hochberg

    # we only correct in the end (treeBH method), leaving here in case someone wants to use my code
    #result_df[f"p_{p_corr_method}"] = corrected_pvals
    #result_df = result_df.sort_values(by=f"p_{p_corr_method}")
    return result_df, f"p_{p_corr_method}"


def chi_squared_test(contingency_table, n, include_expected=False):
    """
    Performs chi square test of independence,
    calling: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html
    """

    chi2, p, dof, expected = chi2_contingency(contingency_table)

    """
    Calculate effect size for the test using Cohen's w effect size
    
    See: https://doi.org/10.3390/math11091982 
    Notably, in a 2 x 2 design, Cohen's w == == phi == Cramer's V == Tschuprow's T == sqrt(chi squared / N).
    However, if it's not a 2 x 2 design (e.g., 2 x 4), this assumption breaks and we need to use Cramer's V. 
    
    References: 
    Cohen, J. (1988). Statistical power analysis for the behavioral sciences (2nd ed.).
    Cramér, H. (1946). Mathematical Methods of Statistics. Princeton University Press.
    """

    # Cohen's w: only if all chi squared tests are 2 x 2
    #w = math.sqrt(chi2 / n)

    # Cramer's V
    r, c = contingency_table.shape  # the number of categories in the first (rows) and second (cols) variables
    k = min(r - 1, c - 1)
    if k <= 0 or n <= 0:
        print(f"something went wrong in the power calculation of the chi squared test on {contingency_table}")
        V = float("nan")
    else:
        V = math.sqrt(chi2 / (n * k))

    result_df = pd.DataFrame({
        "test": ["chi squared"],
        "statistic": [chi2],
        "p": [p],
        "df": [dof],  # Degrees of Freedom
        "effect size": [V],
    })
    if include_expected:
        expected_df = pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns)
        return result_df, expected_df
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


def kmeans_optimal_k(df_pivot, save_path, save_name, k_range=range(2, 10), normalize=False):
    """
    Try different values of k for perform_kmeans, and find the optimal one based on silhouette score.
    """

    records = []  # list of dicts for later 1-SE selection
    results = []  # (k, silhouette_avg) list for compatibility
    result_map = {}  # to retrieve the best_result tuple later

    for k in k_range:
        print(f"\n--- Trying k={k} ---")
        df_result, kmeans_model, centroids, silhouette_avg, sil_se, p_val = perform_kmeans(df_pivot=df_pivot,
                                                                                           clusters=k,
                                                                                           normalize=normalize,
                                                                                           save_path=save_path,
                                                                                           save_name=f"{save_name}_k{k}")
        # silhouette score (already printed and stored inside perform_kmeans)
        silhouette_avg = round(silhouette_score(df_result.drop(columns="Cluster"), df_result["Cluster"]), 2)
        results.append((k, silhouette_avg))

        # calculate cluster sizes
        cluster_sizes = df_result['Cluster'].value_counts().sort_index()
        total_n = len(df_result)

        # print detailed cluster information
        print(f"Cluster sizes:")
        for cluster_id in range(k):
            count = cluster_sizes.get(cluster_id, 0)
            pct = (count / total_n) * 100
            print(f"  Cluster {cluster_id}: {count} ({pct:.2f}%)")

        results.append((k, silhouette_avg))
        result_map[k] = (df_result, kmeans_model, centroids)
        records.append({"k": k, "silhouette": silhouette_avg, "sil_se": sil_se, "p_value": p_val})

    # update optimum [1-SE] selection rule
    sils = np.array([r["silhouette"] for r in records], dtype=float)
    ses = np.array([r["sil_se"] for r in records], dtype=float)
    ks = np.array([r["k"] for r in records], dtype=int)

    idx_star = int(np.nanargmax(sils))
    k_star = int(ks[idx_star])
    sil_star = float(sils[idx_star])
    se_star = float(ses[idx_star]) if np.isfinite(ses[idx_star]) else np.nan

    if np.isfinite(se_star) and se_star > 0.0:
        cutoff = math.ceil((sil_star - se_star) * 100) / 100
        # smallest k whose silhouette >= cutoff
        candidate_indices = np.where(sils >= cutoff)[0]
        k_hat = int(ks[int(candidate_indices[0])]) if candidate_indices.size else k_star
        print(f"\n[1-SE rule] best silhouette at k*={k_star} (sil={sil_star:.2f}, SE={se_star:.2f}) "
              f"→ cutoff={cutoff:.2f} : choose smallest k with sil≥cutoff: k={k_hat}")
    else:
        k_hat = k_star
        print(f"\n[1-SE rule] SE not available or zero at k*={k_star}; defaulting to argmax silhouette.")

    best_k = k_hat
    best_result = result_map[best_k]
    print(f"\nOptimal k (1-SE rule): {best_k}; silhouette score: {dict(results)[best_k]:.6f}")

    return best_k, best_result, results


def perform_kmeans(df_pivot, save_path, save_name, clusters=2, normalize=False, n_iterations=1000,
                   bootstrap_sil_reps=1000, bootstrap_refit=False):
    """
    Perform k-means clustering (scikit-learn's) to group the data into a specified number of clusters.
    We then append the cluster labels to the dataset and calculate the silhouette score,
    which evaluates how well-separated the clusters are. We then test the clustering's statistical significance by
    comparing it to random clusters. This medhod also saves the cluster centroids, which represent the average values
    of features for each cluster.

    If bootstrap_refit=False (default), we *reuse* the fitted clustering and bootstrap
    the rows (labels follow the resample)
    If bootstrap_refit=True, we refit KMeans on each bootstrap sample (slower, more thorough).
    """
    txt_output = list()

    # if normalize is True, normalize the data
    if normalize:
        scaler = StandardScaler()
        df_pivot = scaler.fit_transform(df_pivot)

    # Perform k-means clustering
    # The n_init parameter controls the number of times the KMeans algorithm is run with different centroid seeds; 10 is the default
    kmeans = KMeans(n_clusters=clusters, n_init=10)
    labels = kmeans.fit_predict(df_pivot)
    df_pivot = df_pivot.copy()
    df_pivot.loc[:, "Cluster"] = labels

    # Calculate silhouette score
    """
    The silhouette score measures how similar a data point is to its own cluster compared to other clusters. 
    It ranges from -1 to 1, where a value closer to 1 indicates that the data points are well clustered.
    """
    silhouette_avg = round(silhouette_score(df_pivot.drop(columns="Cluster"), df_pivot["Cluster"]), 2)
    line = f"{clusters}-Means Clustering Silhouette Score: {silhouette_avg:.2f}"
    print(line)
    txt_output.append(line)

    # test the statistical significance of the clustering (against random clusters)
    random_silhouette_scores = [
        silhouette_score(df_pivot.drop(columns="Cluster"), np.random.randint(0, clusters, size=df_pivot.shape[0]))
        for _ in range(n_iterations)
    ]
    p_value = np.mean(silhouette_avg <= np.array(random_silhouette_scores))
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
    n = df_pivot.shape[0]  # the number of observations
    result = list()
    for choice in q_cols:
        # create a contingency table for the current choice and the cluster column
        contingency_table = pd.crosstab(df_pivot["Cluster"], df_pivot[choice])
        # perform the Chi-Square test
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        # Expected: the expected frequencies for each cell in the contingency table, the theoretical frequencies
        # that would occur in each cell of a contingency table if the choices are independent of the cluster
        """
        Calculate effect size for the test using Cohen's w effect size
        Cohen, J. (1988). Statistical power analysis for the behavioral sciences (2nd ed.).

        See also: https://doi.org/10.3390/math11091982 
        Notably, in a 2 x 2 design, Cohen's w == == phi == Cramer's V == Tschuprow's T == sqrt(chi squared / N)

        """
        w = math.sqrt(chi2 / n)
        result.append({"Choice": choice, "Chi2": chi2, "p-value": p, "dof": dof, "effect size": w,"Expected": expected})
    chisq_df = pd.DataFrame(result)
    chisq_df.to_csv(os.path.join(save_path, f"{save_name}_cluster_centroids_chisq.csv"), index=False)

    with open(os.path.join(save_path, f"{save_name}_kmeans_result.txt"), "w") as file:
        for line in txt_output:
            file.write(str(line) + '\n')

    # bootstrap SE for the silhouette (default: no refit)
    sil_se = _silhouette_bootstrap_se(df_pivot, clusters=clusters, B=bootstrap_sil_reps, refit=bootstrap_refit)

    return df_pivot, kmeans, cluster_centroids, float(silhouette_avg), float(sil_se), float(p_value)


def _silhouette_bootstrap_se(df_with_cluster, clusters, B=1000, refit=False):
    """
    Estimate SE of silhouette by bootstrap.
    B: resample rows with replacement and recompute a silhouette score, the SE is then the sample standard deviation/√B
    of those bootstrap silhouette scores. Bigger B = more stable SE but more compute.
    refit: For each bootstrap sample, we reuse the original cluster labels for the resampled rows, then compute silhouette.
    This captures sampling variability of the silhouette given the original partition.
    """
    if B is None or B <= 0:
        return np.nan
    rng = np.random.default_rng()
    X_full = df_with_cluster.drop(columns="Cluster")
    y_full = df_with_cluster["Cluster"].to_numpy()
    n = len(df_with_cluster)

    scores = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        Xb = X_full.iloc[idx, :]
        if not refit:
            yb = y_full[idx]
            # need ≥2 samples per cluster for silhouette
            counts = pd.Series(yb).value_counts()
            if (counts < 2).any():
                continue
            try:
                s = silhouette_score(Xb, yb)
                scores.append(s)
            except Exception:
                continue
        else:
            try:
                km = KMeans(n_clusters=clusters, n_init=10)
                yb = km.fit_predict(Xb)
                if len(np.unique(yb)) < 2:
                    continue
                s = silhouette_score(Xb, yb)
                scores.append(s)
            except Exception:
                continue

    if len(scores) >= 2:
        """
        the standard error of a statistic is estimated by the standard deviation of the bootstrap replicates
        Efron & Tibshirani (1993) describe the bootstrap SE as the sample SD of the bootstrap statistics. 
        That’s the quantity the 1-SE rule should use.
        """
        return float(np.std(scores, ddof=1))
    return np.nan

def plot_cluster_centroids(cluster_centroids, cluster_sems, save_path, save_name, label_map=None, binary=True,
                           threshold=0, overlaid=False, cluster_colors_overlaid=None, fmt="png", label_names_coding=None):
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
            fmt=fmt,
            label_names_coding=label_names_coding
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
                    save_path=save_path,
                    label_names_coding=label_names_coding, fmt="svg"
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
                    save_path=save_path,
                    label_names_coding=label_names_coding
                )
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