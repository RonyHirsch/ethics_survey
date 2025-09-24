import os
os.environ["MPLBACKEND"] = "Agg"   # headless backend, no Tkinter
import re
import logging
import matplotlib
matplotlib.use("Agg", force=True)  # belt-and-suspenders: force Agg to eliminate Tk as it causes redundant runtime errors - BEFORE shap
import shap
import string
import math
from itertools import combinations
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.preprocessing import StandardScaler
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.inspection import permutation_importance
from sentence_transformers import SentenceTransformer
from skbio.stats.distance import permanova
from skbio.stats.distance import DistanceMatrix
import scipy.stats as stats
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind, ttest_1samp, ttest_rel, f_oneway, kruskal, shapiro, levene, norm
from scipy.spatial.distance import cdist
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep
import statsmodels.api as sm
import statsmodels.formula.api as smf
from tqdm import tqdm
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
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
from sklearn.metrics import cohen_kappa_score
from mord import LogisticIT
from statsmodels.stats.multitest import multipletests
import plotter


"""
For free-text response analysis
Sample stopwords (to avoid requiring NLTK)
"""
STOPWORDS = {"the", "and", "to", "of", "a", "is", "in", "that", "it", "as", "for", "on", "with", "this", "be", "or",
             "are", "an", "by", "can", "at", "which", "from", "but", "has", "have", "was", "were", "not", "so",
             "if", "about", "more", "do", "does", "i", "you", "we", "they", "he", "she", "them", "their", "its",
             "being", "will", "would", "should", "there", "some", "what", "when", "how", "why", "just"}

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)


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
                               cv_folds=10, split_test_size=0.3, random_state=42, n_repeats=50,
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=split_test_size,
                                                        random_state=random_state)

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
        ("classifier", RandomForestClassifier(class_weight='balanced', random_state=random_state))  # "balanced" = to deal with having a majority/minority class
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
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
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
        random_state=random_state,
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
        group1 = df[df[group_col] == group1_val][col].dropna()
        group2 = df[df[group_col] == group2_val][col].dropna()
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

    result_df[f"p_{p_corr_method}"] = corrected_pvals
    result_df = result_df.sort_values(by=f"p_{p_corr_method}")
    return result_df, f"p_{p_corr_method}"


def run_ordinal_pipeline(dataframe, dep_col, categorical_cols, order_cols, save_path, save_prefix="",
                         rare_class_threshold=5, n_permutations=1000):
    """
    RandomForestClassifier ➜ mord.LogisticIT() for ordinal modeling
    Accuracy ➜ cohen_kappa_score with weights="quadratic"
    GridSearchCV removed (LogisticIT has only one tunable parameter: alpha)
    Permutation test based on QWK for statistical significance
    """
    df = dataframe.copy()
    df_model = df[categorical_cols + order_cols + [dep_col]].dropna(subset=[dep_col])

    for col in categorical_cols:
        df_model[col] = df_model[col].astype(str)
    rare_merger = RareCategoryMerger(threshold=rare_class_threshold)
    df_model[categorical_cols] = rare_merger.fit_transform(df_model[categorical_cols])

    X = df_model[categorical_cols + order_cols]
    y = df_model[dep_col].astype(int)  # Ensure it's numeric/ordinal

    # Preprocessing
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

    # Full pipeline with ordinal classifier
    ordinal_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticIT())  # Mord's ordinal logistic regression
    ])

    # Cross-validation using Quadratic Weighted Kappa
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    qwk_scores = []
    for train_idx, test_idx in cv.split(X, y):
        ordinal_pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = ordinal_pipeline.predict(X.iloc[test_idx])
        score = cohen_kappa_score(y.iloc[test_idx], preds, weights="quadratic")
        qwk_scores.append(score)

    mean_qwk = np.mean(qwk_scores)

    # Permutation test
    perm_qwk_scores = []
    for i in range(n_permutations):
        y_permuted = shuffle(y, random_state=i).reset_index(drop=True)
        perm_scores = []
        for train_idx, test_idx in cv.split(X, y_permuted):
            ordinal_pipeline.fit(X.iloc[train_idx], y_permuted.iloc[train_idx])
            preds = ordinal_pipeline.predict(X.iloc[test_idx])
            perm_scores.append(cohen_kappa_score(y_permuted.iloc[test_idx], preds, weights="quadratic"))
        perm_qwk_scores.append(np.mean(perm_scores))

    p_value = np.mean([score >= mean_qwk for score in perm_qwk_scores])

    result = {
        'mean_qwk': mean_qwk,
        'cv_scores': qwk_scores,
        'p_value': p_value,
        'dropped_categories': rare_merger.dropped_categories_
    }

    summary_data = {
        'mean_qwk': result['mean_qwk'],
        'p_value': result['p_value'],
        'cv_scores': ', '.join([f"{s:.3f}" for s in result['cv_scores']]),
        'dropped_categories': str(result['dropped_categories'])
    }

    summary_df = pd.DataFrame(list(summary_data.items()), columns=["Metric", "Value"])
    summary_df.to_csv(os.path.join(save_path, f"{save_prefix}_ordinal_summary.csv"), index=False)

    return result




def mixed_effects_model(long_df, dep_col, ind_col1, ind_col2, id_col, cols_to_standardize=list()):
    np.random.seed(42)
    scaler = StandardScaler()
    if len(cols_to_standardize) > 0:
        long_df[cols_to_standardize] = scaler.fit_transform(long_df[cols_to_standardize])
    # Fit a mixed-effects model: dep_col ~ ind_col1 + ind_col2 + (1 | id_col)
    model = smf.mixedlm(f"{dep_col} ~ {ind_col1} + {ind_col2}", data=long_df, groups=id_col, re_formula="~1")
    result = model.fit()
    summary = result.summary().as_text()

    # fixed effect coefficients
    fixed_feature_weights = result.fe_params

    result_df = pd.DataFrame({
        "Coefficient": result.fe_params.index
    })
    result_df["Value"] = result.fe_params.values
    result_df["Standard Error"] = result.bse.loc[result.fe_params.index].values  # Align lengths
    result_df["T-Statistic"] = result.tvalues.loc[result.fe_params.index].values
    result_df["p-value"] = result.pvalues.loc[result.fe_params.index].values

    # Residuals for each observation
    residuals_df = pd.DataFrame({
        "Observation": long_df.index,
        "Residual": result.resid,
        "Fitted Value": result.fittedvalues,
        dep_col: long_df[dep_col]
    })

    # Variance explained
    # Marginal R²: Fixed effects only
    fixed_effect_variance = result.rsquared if hasattr(result, 'rsquared') else None

    # Conditional R²: Fixed + random effects
    random_variance = result.cov_re.iloc[0, 0]  # Random effect variance
    residual_variance = result.scale  # Residual variance
    conditional_r2 = random_variance / (random_variance + residual_variance)
    r2_df = pd.DataFrame({
        "Metric": ["Marginal R²", "Conditional R²"],
        "Value": [fixed_effect_variance, conditional_r2]
    })

    # Descriptive statistics grouped by ind_col2
    descriptive_stats = long_df.groupby(ind_col2)[[dep_col, ind_col1]].describe().reset_index()

    # Convert model summary to DataFrame for saving
    summary_df = pd.DataFrame({
        "Summary": summary.split("\n")
    })

    """
    Posthoc
    """
    # Welch's t-test for post-hoc analysis (ind_col2 must have exactly 2 levels)
    if long_df[ind_col2].nunique() == 2:
        group_0 = long_df[long_df[ind_col2] == 0][dep_col]
        group_1 = long_df[long_df[ind_col2] == 1][dep_col]
        t_stat, p_value = ttest_ind(group_0, group_1, equal_var=False)
        posthoc_df = pd.DataFrame({
            "Test": ["Welch's t-test"],
            "Statistic": [t_stat],
            "P-Value": [p_value],
            "Interpretation": ["Significant" if p_value < 0.05 else "Not Significant"]
        })
    else:
        posthoc_df = pd.DataFrame({
            "Error": ["Post-hoc test requires exactly 2 levels in ind_col2"]
        })

    return result_df, residuals_df, r2_df, summary_df, descriptive_stats, posthoc_df


def permanova_on_pairwise_distances(data, columns, group_col, dist_metric="euclidean", perm_num=1000,
                                    multicomp_method="bonferroni"):
    """
    PERMANOVA  (Permutational Multivariate Analysis of Variance) is a non-parametric method that tests whether two
    or more groups whether are significantly different based on a categorical factor.
    Here, the categorical factor we choose is a DISTANCE metric.
    It is conceptually similar to ANOVA except that it operates on a distance matrix,
    which allows for multivariate analysis. PERMANOVA computes a pseudo-F statistic.
    https://scikit.bio/docs/dev/generated/skbio.stats.distance.permanova.html

    It compares the variance of distances (e.g., Euclidean, Manhattan) within and between groups.
    We will treat each person (=row) as a vector (data[sub, columns]),
    compute the distance between all people's vectors,
    and test if the distances between groups (="by") are significantly larger than within each group.

    Statistical significance is assessed via a permutation test.
    Unlike PCA + MANOVA, this method doesn't assume linear relationships and is robust to outliars.
    """

    relevant_data = data[columns]
    groups = data[group_col]
    # pairwise distances
    distance_matrix = pairwise_distances(relevant_data, metric=dist_metric)
    # convert to DistanceMatrix: https://scikit.bio/docs/dev/generated/skbio.stats.distance.permanova.html
    distance_matrix = DistanceMatrix(distance_matrix, ids=relevant_data.index.astype(str))
    # permanova time
    permanova_results = permanova(distance_matrix, grouping=list(groups), permutations=perm_num)
    # calculate dof
    df_between = groups.nunique() - 1
    df_residual = len(groups) - groups.nunique()
    # save
    result_df = pd.DataFrame({
        "test": [permanova_results["test statistic name"]],
        "statistic": [permanova_results["test statistic"]],
        "p": [permanova_results["p-value"]],
        "df": [f"({df_between}, {df_residual})"],
        "N": [permanova_results["sample size"]],
        "Groups": [permanova_results["number of groups"]]
    })


    """
    Post Hoc - which ratings are the most likely to contribute to the difference between the groups?
    Perform a one-way ANOVA  
    """
    clusters = data[group_col].unique()
    anova_results = list()
    descriptives = list()

    for column in relevant_data.columns:
        groups = [relevant_data[data[group_col] == cluster][column] for cluster in data[group_col].unique()]
        # assumption checks
        normality_pvals = [shapiro(group)[1] for group in groups if len(group) > 3]  # only test if group size > 3
        variance_pval = levene(*groups)[1] if len(groups) > 1 else 1  # Levene's test for homogeneity of variance
        # if anova assumptions don't break
        if all(p > 0.05 for p in normality_pvals) and variance_pval > 0.05:
            # Perform ANOVA
            stat, p = f_oneway(*groups)
            test_type = "ANOVA"
            df_between = len(clusters) - 1
            df_residual = len(data) - len(clusters)
            degrees_of_freedom = f"({df_between}, {df_residual})"
        else:  # assumptions break, do a Kruskal-Wallis
            stat, p = kruskal(*groups)
            test_type = "Kruskal-Wallis"
            degrees_of_freedom = f"{len(clusters) - 1}"
        # summarize results
        anova_results.append({
            "feature": column,
            "test": test_type,
            "statistic": stat,
            "p-value": p,
            "df": degrees_of_freedom,
        })
        # add interpretation:
        descriptive_stats = relevant_data.groupby(data[group_col])[column].agg(["mean", "std"]).reset_index()
        descriptive_stats = descriptive_stats.rename(columns={group_col: "group"})
        descriptive_stats["feature"] = column
        descriptives.append(descriptive_stats)

    anova_df = pd.DataFrame(anova_results)
    # correct p-value for multiple comparisons
    anova_df[f"p {multicomp_method}"] = multipletests(anova_df["p-value"], method=multicomp_method)[1]
    anova_df.sort_values(f"p {multicomp_method}", ascending=True, inplace=True)
    descriptives_df = pd.concat(descriptives, ignore_index=True)

    return result_df, anova_df.reset_index(inplace=False, drop=True), descriptives_df


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


def kruskal_wallis_test(merged_df=None, group_col=None, ordinal_col=None, grouped=None):
    if merged_df and group_col and ordinal_col:
        grouped = merged_df.groupby(group_col)[ordinal_col].apply(list)
    else:  # we have "grouped" instead
        pass
    ks_stat, p_value = kruskal(*grouped)
    result_df = pd.DataFrame({
        "test": ["kruskal-wallis"],
        "statistic": [ks_stat],
        "p": [p_value],
        "df": [len(grouped) - 1]  # dof in kruskal wallis is k-1 where k is the number of indpendent groups
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


def dependent_samples_ttest(list_group1, list_group2, ci=0.95):
    ttest_result = ttest_rel(list_group1, list_group2)
    conf_interval = ttest_result.confidence_interval(confidence_level=ci)
    result_df = pd.DataFrame({
        "test": ["paired samples t-test"],
        "statistic": [ttest_result.statistic],
        "p": [ttest_result.pvalue],
        "df": [ttest_result.df],
        f"{ci} CI low": [conf_interval.low],
        f"{ci} CI high": [conf_interval.high]
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
    pca = PCA(n_components=components)  # deterministic by default
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


def kmeans_optimal_k(df_pivot, save_path, save_name, k_range=range(2, 10), normalize=False):
    """
    Try different values of k for perform_kmeans, and find the optimal one based on silhouette score.
    """
    best_score = -1
    best_k = None
    best_result = None

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
                   bootstrap_sil_reps=1000, bootstrap_refit=False, random_state=42):
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
    kmeans = KMeans(n_clusters=clusters, random_state=random_state, n_init=10)
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
    sil_se = _silhouette_bootstrap_se(df_pivot, clusters=clusters, B=bootstrap_sil_reps, refit=bootstrap_refit,
                                      rng_seed=random_state)

    return df_pivot, kmeans, cluster_centroids, float(silhouette_avg), float(sil_se), float(p_value)


def _silhouette_bootstrap_se(df_with_cluster, clusters, B=1000, refit=False, rng_seed=42):
    """
    Estimate SE of silhouette by bootstrap.
    B: resample rows with replacement and recompute a silhouette score, the SE is then the sample standard deviation/√B
    of those bootstrap silhouette scores. Bigger B = more stable SE but more compute.
    refit: For each bootstrap sample, we reuse the original cluster labels for the resampled rows, then compute silhouette.
    This captures sampling variability of the silhouette given the original partition.
    """
    if B is None or B <= 0:
        return np.nan
    rng = np.random.default_rng(rng_seed)
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
                km = KMeans(n_clusters=clusters, random_state=42, n_init=10)
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


def two_proportion_ztest(group1, df1, group2, df2, col_items, col_prop, col_n):

    # align dfs by items
    df1 = df1.copy().set_index(col_items)
    df2 = df2.copy().set_index(col_items)
    df2 = df2.reindex(df1.index)

    result = []

    for item in df1.index:
        # Get counts for the item
        count1 = round(df1.loc[item, col_prop] / 100 * df1.loc[item, col_n])
        count2 = round(df2.loc[item, col_prop] / 100 * df2.loc[item, col_n])

        # Total number of observations in each group
        nobs1 = df1.loc[item, col_n]
        nobs2 = df2.loc[item, col_n]

        # Perform the two-proportion z-test
        stat, p_value = proportions_ztest([count1, count2], [nobs1, nobs2])

        # Append the results
        result.append({
            "Item": item,
            f"Proportion {group1}": df1.loc[item, col_prop],
            f"Proportion {group2}": df2.loc[item, col_prop],
            "z-statistic": stat,
            "p-value": p_value,
        })

    # Return the results as a DataFrame
    return pd.DataFrame(result)


def preprocess_text(text):
    """
    Preprocess text for analysis: lowercase, remove punctuation, remove stopwords
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    words = text.split()  # tokenization (split by spaces)
    words = [word for word in words if word not in STOPWORDS]  # remove stopwords
    return " ".join(words)  # cleaned text


def preprocess_text_basic(text):
    """
    Does NOT remove stop words etc; just lowercase and remove extra spaces
    :param text:
    :return:
    """
    return str(text).strip().lower()


def find_optimal_topics(df, text_column, topic_range=range(2, 10)):
    """
    ***
    WHY I ENDED UP NOT USING IT:
    Despite its great results on medium or large sized texts (>50 words), typically mails and news articles are about
    this size range, LDA poorly performs on short texts like Tweets, Reddit posts or StackOverflow titles’ questions.
    ***
    The most popular Topic Modeling algorithm is LDA, Latent Dirichlet Allocation (LDA).
    - Latent because the topics are “hidden”. We have a bunch of texts and we want the algorithm to put them into
    clusters that will make sense to us.
    - Dirichlet stands for the Dirichlet distribution the model uses as a prior to generate document-topic and
    word-topic distributions.
    - Allocation because we want to allocate topics to our texts.
    We will choose the optimal number of topics for LDA analysis based on the coherence score.
    Coherence Score measures how "interpretable" topics are by checking word co-occurrence
    (Higher coherence = better topic separation).
    We will use Gensim to calculate coherence.
    """

    tokenized_text = df[text_column].dropna().astype(str).tolist()
    tokenized_text = [text.split() for text in tokenized_text]

    dictionary = Dictionary(tokenized_text)
    corpus = [dictionary.doc2bow(text) for text in tokenized_text]

    coherence_scores = []

    for num in topic_range:
        lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num, passes=10, random_state=42)
        """
        Coherence Score measures how well words in a topic make sense together. 
        It evaluates if words frequently appear together in actual responses.
        For dyads of words in a given topic, it calculates the Pointwise Mutual Information (how often the words
        actually co-occurred in responses). 
        """
        coherence_model = CoherenceModel(model=lda_model, texts=tokenized_text, dictionary=dictionary, coherence='c_v')
        coherence_scores.append(coherence_model.get_coherence())

    # If we want, we can plot the coherence scores (THE HIGHER THE BETTER)
    #import matplotlib.pyplot as plt
    #plt.plot(topic_range, coherence_scores, marker='o')
    #plt.xlabel("Number of Topics")
    #plt.ylabel("Coherence Score")
    #plt.title("Optimal Number of Topics")
    #plt.show()

    # best number of topics based on coherence score
    best_num_topics = topic_range[coherence_scores.index(max(coherence_scores))]
    print(f"\nOptimal number of topics: {best_num_topics}, coherence_score = {max(coherence_scores)}")
    return best_num_topics


def topic_modelling_LDA(df, text_column, save_path, save_name, num_topics=None):
    """
    I assume df contains a row per repsonse (participant), with an ID column, and some column where all the free
    text is
    ***
    WHY I ENDED UP NOT USING IT:
    Despite its great results on medium or large sized texts (>50 words), typically mails and news articles are about
    this size range, LDA poorly performs on short texts like Tweets, Reddit posts or StackOverflow titles’ questions.
    It doesn't have context-awareness (the ability to detect that a meaning of a word can change based on context)
    ***
    The most popular Topic Modeling algorithm is LDA, Latent Dirichlet Allocation (LDA).
    - Latent because the topics are “hidden”. We have a bunch of texts and we want the algorithm to put them into
    clusters that will make sense to us.
    - Dirichlet stands for the Dirichlet distribution the model uses as a prior to generate document-topic and
    word-topic distributions.
    - Allocation because we want to allocate topics to our texts.
    """
    # preprocess
    df["processed_text"] = df[text_column].astype(str).apply(preprocess_text)
    logging.info("Preprocessing done")

    """
    Word Frequency Analysis
    """
    all_words = " ".join(df["processed_text"]).split()
    word_freq = Counter(all_words)
    total_words = sum(word_freq.values())
    word_freq_df = pd.DataFrame(word_freq.items(), columns=["Word", "Frequency"])
    word_freq_df["Proportion (%)"] = (word_freq_df["Frequency"] / total_words) * 100
    word_freq_df.sort_values(by="Frequency", ascending=False, inplace=True)
    word_freq_csv_path = os.path.join(save_path, f"{save_name}_word_freqs.csv")
    word_freq_df.to_csv(word_freq_csv_path, index=False)
    logging.info("Word Frequency Analysis done")
    """
    Topic modeling with LDA
    """
    if num_topics is None:
        num_topics = find_optimal_topics(df=df, text_column="processed_text")
    logging.info("Found optimal number of topics")

    tokenized_text = df["processed_text"].dropna().astype(str).tolist()  # avoids issues with NaN values or non-string inputs
    tokenized_text = [text.split() for text in tokenized_text]
    dictionary = Dictionary(tokenized_text)
    corpus = [dictionary.doc2bow(text) for text in tokenized_text]
    logging.info("Created the corpus")

    """
    LDA is a probabilistic generative model for discovering hidden topics.
    Each response is considered a mixture of topics. Each topic is a mixture of words, with some words more important 
    than others.
    LDA assigns probabilities of topics to responses and words to topics, and tries to learn these distributions to find 
    coherent themes in the data.
    """
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=10, random_state=42)
    logging.info("LDA model trained")

    # extract the topics
    topics_data = []
    for topic_idx, topic in enumerate(lda.show_topics(num_topics=num_topics, formatted=False)):
        logging.info(f"Extracting topics: {topic_idx}")
        top_words = [word for word, _ in topic[1]]
        topics_data.append({"Topic": topic_idx + 1, "Top Words": ", ".join(top_words)})
    topics_df = pd.DataFrame(topics_data)
    logging.info("Done; created topic df")
    topics_csv_path = os.path.join(save_path, f"{save_name}_topics.csv")
    topics_df.to_csv(topics_csv_path, index=False)

    return


def topic_modelling_bertopic(df, text_column, save_path, save_name, use_tfidf=True):
    """
    Perform topic modeling using BERTopic (transformer-based architecture, enabling context-aware analysis of text)
    https://arxiv.org/abs/2203.05794
     M. Grootendorst, “BERTopic: Neural topic modeling with a class-based TF-IDF procedure,” CoRR, vol. /2203.05794, 2022

    - `use_tfidf`: If True, uses TF-IDF representation. If False, uses CountVectorizer.
    """

    # BASIC - just lowercase and removing extra spaces
    df["processed_text"] = df[text_column].astype(str).apply(preprocess_text_basic)
    texts = df["processed_text"].tolist()
    logging.info("Preprocessing completed")

    """
    Create word embeddings:
    SentenceTransformer = a wrapper around pretrained BERT-like models that generate sentence embeddings.
    Normally, BERT models process individual words and do not generate a fixed-size embedding for an entire sentence.
    SentenceTransformer solves this by using models fine-tuned for sentence similarity tasks (like SBERT).
    Instead of getting embeddings per word, we get a single vector per sentence (=response). 
    """
    # this is a compressed and faster version of BERT (fewer params and runs w/o GPU)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # outputs a 384-dim vector for each response, was trained to capture semantic similarities
    logging.info("Sentence Transformer loaded.")

    """
    Text Vectorization:
    
    - CountVectorizer(): Converts text into a bag-of-words representation. It creates a response-term matrix where each 
    row is a sentence, and each column is a word: each cell counts how many times a word appears in a response.
    It works well for keyword frequency analysis but is highly affected by common words and doesn't handle word importance
    
    - TfidfVectorizer(): applies TF-IDF weighting, so instead of raw word counts it assigns importance scores to words:
    TF−IDF= TF×log(N/DF). Such that:
        TF (Term Frequency): How often a word appears in a document.
        DF (Document Frequency): How many documents contain the word. (responses)
        N: Total number of documents. (responses)
    
    """
    vectorizer = TfidfVectorizer() if use_tfidf else CountVectorizer()
    method = "TF-IDF" if use_tfidf else "CountVectorizer"
    logging.info(f"Vectorization method selected: {method}")

    """
    Fit BERTopic model to the data: it will use Uses HDBSCAN (Hierarchical Density-Based Clustering) to cluster 
    similar embeddings. BERTopic Assigns each sentence to a topic based on closeness in embedding space.
    """
    topic_model = BERTopic(
        embedding_model=embedding_model,  # BERT embeddings: sentences are turned to vectors
        vectorizer_model=vectorizer,  # TF-IDF or CountVectorizer
        min_topic_size=5,  # Minimum cluster size
        verbose=True
    )
    topics, probs = topic_model.fit_transform(texts)  # fits the model to the dataset and returns topic labels.
    logging.info("BERTopic model trained.")

    # save topic information
    topics_df = topic_model.get_topic_info()
    topics_df.to_csv(os.path.join(save_path, f"{save_name}_topics.csv"), index=False)

    # save topics for each response
    doc_topics = pd.DataFrame({"Text": texts, "Topic": topics})
    doc_topics.to_csv(os.path.join(save_path, f"{save_name}_document_topics.csv"), index=False)

    logging.info("Topics saved")

    # Show top words per topic
    print("\nTop words per topic:")
    print(topic_model.get_topic_info())

    # Save visualization
    topic_model.visualize_barchart().write_html(os.path.join(save_path, f"{save_name}_topic_chart.html"))
    logging.info("Visualization saved")

    return topic_model


def preprocess_text_gsdmm(text):
    """
    - Tokenization using spaCy tokenizer.
    - Removing stop words and 1 character words.
    - Stemming using the nltk library’s stemmer.
    - Removing empty responses and responses with more than 30 tokens.
    - Removing unique token (with a term frequency = 1).
    """
    text = text.lower()  # lowercase
    text = re.sub(r'\d +', "", text)  # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = text.strip()  # remove whitespace
    # tokenization
    nlp = English()
    tokens = nlp(text)
    words = text.split()  # tokenization (split by spaces)
    words = [word for word in words if word not in STOPWORDS]  # remove stopwords
    return


def topic_modelling_GSDMM(df, text_column, save_path, save_name):
    """
    The Gibbs Sampling Dirichlet Mixture Model (GSDMM) is an “altered” LDA algorithm, showing great results on short
    texts, that makes the initial assumption: 1 topic corresponds to 1 document.
    The words within a document are generated using the same unique topic, and not from a mixture of topics as it was
    in the original LDA.
    :return:
    """

    # preprocess
    df["processed_text"] = df[text_column].astype(str).apply(preprocess_text_gsdmm)

    return


def premutations_for_array(matrix, metric_func, n_permutations=1000, print_iter=False, *args, **kwargs):
    """
    Performs permutation test on a matrix using a specified metric function.

    Parameters:
    - matrix (ndarray): The input data matrix (e.g., raters × items).
    - metric_func (callable): Function that takes the matrix (and optional args) and returns a score.
    - n_permutations (int): Number of permutations.
    - print_iter (bool): Whether to print progress.
    - *args, **kwargs: Additional arguments passed to metric_func.

    Returns:
    - null_scores (list): List of scores from permuted matrices.
    """
    null_alphas = []
    for i in range(n_permutations):
        if print_iter:
            print(f"iter {i}")
        shuffled_matrix = np.apply_along_axis(np.random.permutation, axis=0, arr=matrix)  # Shuffle columns
        shuffled_matrix = np.apply_along_axis(np.random.permutation, axis=1, arr=shuffled_matrix)  # Shuffle rows
        alpha_perm = metric_func(shuffled_matrix, *args, **kwargs)
        null_alphas.append(alpha_perm)
    return np.array(null_alphas)


def nominal_metric(a, b):
    return a != b


def interval_metric(a, b):
    return (a - b) ** 2


def ratio_metric(a, b):
    return ((a - b) / (a + b)) ** 2


def krippendorff_alpha(data, metric=interval_metric, convert_items=float):
    """
    *** CREDIT TO: https://github.com/grrrr/krippendorff-alpha/blob/master/krippendorff_alpha.py ***
    I changed some stuff for simplicity: The original version supports both a list of dicts and a matrix.
    In my case, I know I only have matrices, so I made this streamlined for matrix input only. This avoids
    extra logic for generality that I don't need - AND IT MAKES THE PERMUTATION TEST ***FASTER***!!!!!

    Calculate Krippendorff's alpha (inter-rater reliability):

    data is in the format
    [
        {unit1:value, unit2:value, ...},  # coder 1
        {unit1:value, unit3:value, ...},   # coder 2
        ...                            # more coders
    ]
    or
    it is a sequence of (masked) sequences (list, numpy.array, numpy.ma.array, e.g.) with rows corresponding to coders and columns to items

    metric: function calculating the pairwise distance
    force_vecmath: force vector math for custom metrics (numpy required)
    convert_items: function for the type conversion of items (default: float)
    missing_items: indicator for missing items (default: None)
    """

    # number of coders
    m = len(data)

    # convert input data to a dict of items
    units = {}
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            try:
                its = units[j]
            except KeyError:
                its = []
                units[j] = its
            its.append(convert_items(val))

    units = {k: v for k, v in units.items() if len(v) > 1}  # units with pairable values
    n = sum(len(pv) for pv in units.values())  # number of pairable values

    if n == 0:
        raise ValueError("No items to compare.")

    Do = 0.
    for v in units.values():
        v = np.asarray(v)
        Du = sum(np.sum(metric(v, vi)) for vi in v)
        Do += Du / float(len(v) - 1)
    Do /= float(n)
    if Do == 0:
        return 1.
    De = 0.
    all_vals = list(units.values())
    for v1 in all_vals:
        v1 = np.asarray(v1)
        for v2 in all_vals:
            De += sum(np.sum(metric(v1, vi)) for vi in v2)
    De /= float(n * (n - 1))
    return 1. - Do / De if (Do and De) else 1.


def multinom_logistic_regression(df, id_col, categorical_dep_col, binary_feature_col, save_path, save_name,
                                 reference_category=None, use_class_weights=True):

    multinom_df = df.copy()
    multinom_df = multinom_df.loc[:, [id_col, binary_feature_col, categorical_dep_col]]
    multinom_df.to_csv(os.path.join(save_path, f"{save_name}_multinom_df.csv"), index=False)

    # set reference category if specified
    if reference_category is not None:
        if not isinstance(multinom_df[categorical_dep_col].dtype, pd.CategoricalDtype):
            multinom_df[categorical_dep_col] = multinom_df[categorical_dep_col].astype('category')
        if not multinom_df[categorical_dep_col].cat.ordered:
            multinom_df[categorical_dep_col] = multinom_df[categorical_dep_col].cat.as_ordered()
        if reference_category not in multinom_df[categorical_dep_col].cat.categories:
            raise ValueError(f"Reference category '{reference_category}' not found in '{categorical_dep_col}'")
        multinom_df[categorical_dep_col] = multinom_df[categorical_dep_col].cat.reorder_categories(
            [reference_category] + [cat for cat in multinom_df[categorical_dep_col].cat.categories if
                                    cat != reference_category],
            ordered=True
        )

    # prepare X (predictor) and y (dep)
    """
    Preparing the X: adding an intercept term (a constant column of 1s). This is necessary because regression models
    include an intercept term by default, and statsmodels expects you to add it explicitly. 
    If you don’t add the constant, the model will assume the intercept is 0, which usually leads to a misfit model 
    unless you're modeling something that naturally goes through the origin (which is rare).
    """
    X = sm.add_constant(multinom_df[binary_feature_col])  # adds a column of 1s as an intercept.
    y = multinom_df[categorical_dep_col]

    """
    If we want to use class weighting because we want to infer relationships, not to classify future observations
    And we do, because we are checking if the binary feature predicts class 
    """
    weights = None
    if use_class_weights:
        class_counts = multinom_df[categorical_dep_col].value_counts()
        total = class_counts.sum()
        weights = multinom_df[categorical_dep_col].map(lambda c: total / class_counts[c])

    # fit model
    model = sm.MNLogit(y, X)
    result = model.fit(disp=False, weights=weights)  # disp=False: don't present the iterations

    # predicted class
    predicted_probs = result.predict(X)
    predicted_class = predicted_probs.idxmax(axis=1)
    """
    Accuracy score: if 0 then it means the model learns nothing useful from the binary feature, 
    predicts the same group (reference) every time, and non of those predictions match the actual labels. 
    """
    accuracy = accuracy_score(y, predicted_class)

    # confidence intervals
    conf_int = result.conf_int()
    conf_int.columns = ['ci_lower', 'ci_upper']
    conf_int_odds = np.exp(conf_int)

    """
    Likelihood Ratio Test (LRT): compare this model to the null model, to test whether the binary predictor provides 
    useful explanatory power beyond random guessing or the base rates of categories.
    """
    # fit null model (intercept only)
    X_null = pd.DataFrame({'const': 1}, index=multinom_df.index)
    null_model = sm.MNLogit(y, X_null)
    null_result = null_model.fit(disp=False, weights=weights)  # use the same weights in the null model
    # calculate LRT
    lr_stat = 2 * (result.llf - null_result.llf)
    df_diff = result.df_model - null_result.df_model
    p_value_lr = stats.chi2.sf(lr_stat, df_diff)

    with open(os.path.join(save_path, f"{save_name}_multinom_logistic_regression_summary.txt"), "w") as f:
        print("=== Multinomial Logistic Regression Summary ===\n", file=f)
        print(result.summary(), file=f)
        print(f"\nReference category: {y.cat.categories[0]}", file=f)

        print("\n=== Odds Ratios ===\n", file=f)
        print(np.exp(result.params), file=f)

        print("\n=== P-values ===\n", file=f)
        print(result.pvalues, file=f)

        print("\n=== 95% Confidence Intervals (Beta Coefficients) ===\n", file=f)
        print(conf_int, file=f)

        print("\n=== 95% Confidence Intervals (Odds Ratios) ===\n", file=f)
        print(conf_int_odds, file=f)

        print(f"\n=== Classification Accuracy ===\n{accuracy:.4f}", file=f)

        print("\n=== Likelihood Ratio Test vs Null Model ===\n", file=f)
        print(f"LR statistic: {lr_stat:.4f}", file=f)
        print(f"Degrees of freedom: {df_diff}", file=f)
        print(f"P-value: {p_value_lr:.4g}", file=f)

        if use_class_weights:
            print("\nClass weighting was applied to correct for imbalance.", file=f)
        else:
            print("\nClass weighting was NOT applied.", file=f)

    # summary df
    params = result.params
    odds_ratios = np.exp(params)
    p_values = result.pvalues
    category_labels = y.cat.categories[1:]  # excludes reference
    summary_table = pd.DataFrame({
        f"category (vs. {y.cat.categories[0]})": category_labels,
        "estimate (beta)": params.xs(binary_feature_col, axis=0),
        "odds_ratio": odds_ratios.xs(binary_feature_col, axis=0),
        "p_value": p_values.xs(binary_feature_col, axis=0)
    })

    model_stats = pd.DataFrame({
        f"category (vs. {y.cat.categories[0]})": ["---"],
        "estimate (beta)": [np.nan],
        "odds_ratio": [np.nan],
        "p_value": [np.nan]
    })

    model_stats_extra = pd.DataFrame({
        f"category (vs. {y.cat.categories[0]})": [
            "Classification Accuracy",
            "LR stat vs Null Model",
            "LR p-value",
            "Class weighting applied"
        ],
        "estimate (beta)": [accuracy, lr_stat, p_value_lr, int(use_class_weights)],
        "odds_ratio": [np.nan] * 4,
        "p_value": [np.nan] * 4
    })

    full_summary = pd.concat([summary_table, model_stats, model_stats_extra], ignore_index=True)
    full_summary.to_csv(os.path.join(save_path, f"{save_name}_multinom_logistic_regression_summary.csv"), index=False)

    clean_summary = pd.DataFrame({
        "Outcome Group": category_labels,
        "Coef (beta)": params.xs(binary_feature_col, axis=0).round(3),
        "p-value": p_values.xs(binary_feature_col, axis=0).round(3),
        "95% CI (beta)": [
            f"[{lo:.3f}, {hi:.3f}]"
            for lo, hi in zip(conf_int.xs(binary_feature_col, level=1)['ci_lower'],
                              conf_int.xs(binary_feature_col, level=1)['ci_upper'])
        ],
        "Odds Ratio": odds_ratios.xs(binary_feature_col, axis=0).round(3),
        "95% CI (OR)": [
            f"[{lo:.3f}, {hi:.3f}]"
            for lo, hi in zip(conf_int_odds.xs(binary_feature_col, level=1)['ci_lower'],
                              conf_int_odds.xs(binary_feature_col, level=1)['ci_upper'])
        ],
    })

    clean_summary.sort_values("p-value", inplace=True)
    clean_summary.to_csv(os.path.join(save_path, f"{save_name}_multinom_logistic_regression_feature_summary.csv"), index=False)

    return



