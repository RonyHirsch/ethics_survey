import os
import re
import logging
import shap
import pandas as pd
import numpy as np
import string
from collections import Counter
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.preprocessing import StandardScaler
import transformers
import spacy
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from skbio.stats.distance import permanova
from skbio.stats.distance import DistanceMatrix
from skbio.stats.ordination import pcoa
from kmodes.kprototypes import KPrototypes
import scipy.stats as stats
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind, ttest_1samp, ttest_rel, f_oneway, kruskal, shapiro, levene
from scipy.spatial.distance import cdist
from sklearn.utils import shuffle
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.multivariate.manova import MANOVA
from tqdm import tqdm
import time
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import norm
import statsmodels.api as sm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.utils import resample
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
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
                               cv_folds=10):

    print("starting RF pipeline")
    # dep_col is assumed to be binary
    df = dataframe.copy()
    df_model = df[categorical_cols + order_cols + [dep_col]].dropna(subset=[dep_col])

    # Merge rare categories
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

    # Preprocessing
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

    # Full pipeline
    rf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(class_weight='balanced', random_state=42))  # "balanced" = to deal with having a majority/minority class
    ])

    # Grid search for hyperparameter tuning
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_features': ['sqrt', 'log2']
    }
    print("grid search for hyperparameter tuning")
    grid_search = GridSearchCV(rf_pipeline, param_grid, cv=cv_folds, scoring=scoring_method, n_jobs=-1)
    grid_search.fit(X, y)

    # Cross-validation with best estimator
    best_model = grid_search.best_estimator_
    print(f"Grid search complete. Best params: {grid_search.best_params_}")

    print("cross-validation with best model")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(best_model, X, y, cv=cv, scoring=scoring_method)
    mean_scores = scores.mean()
    print(f"mean {scoring_method}: {mean_scores:.4f}")

    # Permutation test
    print(f"permutation test with {n_permutations} iterations")
    perm_accuracies = []
    for i in tqdm(range(n_permutations), desc="Running permutation test"):
        y_permuted = shuffle(y, random_state=i).reset_index(drop=True)
        perm_scores = []
        for train_idx, test_idx in cv.split(X, y_permuted):
            best_model.fit(X.iloc[train_idx], y_permuted.iloc[train_idx])
            preds = best_model.predict(X.iloc[test_idx])
            if scoring_method == "accuracy":
                perm_scores.append(accuracy_score(y_permuted.iloc[test_idx], preds))
            elif scoring_method == "f1":
                perm_scores.append(f1_score(y_permuted.iloc[test_idx], preds))
        perm_accuracies.append(np.mean(perm_scores))

    p_value = np.mean([acc >= mean_scores for acc in perm_accuracies])
    print(f"permutation test complete. p-value: {p_value:.4f}")

    df_model.to_csv(os.path.join(save_path, f"{save_prefix}_df_model.csv"), index=False)

    # Feature importances
    print("extracting feature importances")
    best_model.fit(X, y)
    feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
    importances = best_model.named_steps['classifier'].feature_importances_
    importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance',
                                                                                                     ascending=False)

    """
    Plot directionality
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    for col in order_cols:
        # Stacked bar: cluster distribution by experience rating
        ct = pd.crosstab(df_model[col], df_model[dep_col], normalize='index')
        fig1 = ct.plot(kind='bar', stacked=True, colormap='tab10').get_figure()
        plt.title(f"Cluster Proportion by '{col}' Rating")
        plt.xlabel(col)
        plt.ylabel("Proportion in Each Cluster")
        fig1.savefig(os.path.join(save_path, f"{col}_cluster_distribution"))
        plt.close(fig1)
        ct.to_csv(os.path.join(save_path, f"{col}_cluster_distribution.csv"))

        # Mean experience by cluster
        fig2, ax2 = plt.subplots()
        sns.pointplot(data=df_model, x=dep_col, y=col, ci='sd', ax=ax2)
        ax2.set_title(f"Mean '{col}' by Cluster")
        ax2.set_ylim(1, 5)
        fig2.savefig(os.path.join(save_path, f"{col}_mean_by_cluster"))
        plt.close(fig2)
        df_model.loc[:, [dep_col, col]].to_csv(os.path.join(save_path, f"{col}_cluster_distribution.csv"))

    """
    SHAP (SHapley Additive exPlanations) analysis is a method for interpreting the predictions of machine learning 
    models by explaining how each feature contributes to the model's output.
    https://shap.readthedocs.io/en/latest/
    """
    import matplotlib.pyplot as plt

    print("running SHAP analysis")
    X_transformed = best_model.named_steps['preprocessor'].transform(X)

    # convert sparse matrix to dense if needed
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
    feature_names = np.array(feature_names).flatten()

    explainer = shap.TreeExplainer(best_model.named_steps['classifier'])
    shap_values = explainer.shap_values(X_transformed)
    shap_values_to_plot = shap_values[1] if isinstance(shap_values, list) and len(shap_values) == 2 else shap_values
    assert shap_values_to_plot.shape[1] == len(feature_names), (
        f"SHAP feature count ({shap_values_to_plot.shape[1]}) does not match "
        f"feature_names length ({len(feature_names)})"
    )


    """
    SHAP summary plot
    Each dot = one observation
    X-axis = SHAP value: How much that feature pushed the prediction toward one class or the other.
    Negative SHAP = push toward class 0
    Positive SHAP = push toward class 1
    Color = feature value (red = high, blue = low)
    Y-axis = features ranked by importance (from top to bottom)
    """
    shap.summary_plot(shap_values_to_plot, X_transformed, feature_names=feature_names, show=False)
    shap_plot_path = os.path.join(save_path, f"{save_prefix}_shap_summary_plot_{scoring_method}.png")
    plt.tight_layout()
    plt.savefig(shap_plot_path)
    plt.close()

    # Save mean absolute SHAP values
    mean_abs_shap = np.mean(np.abs(shap_values_to_plot), axis=0).flatten()
    mean_abs_shap.to_csv(os.path.join(save_path, f"{save_prefix}_shap_importances_{scoring_method}.csv"), index=False)

    # summary
    result = {
        'best_params': grid_search.best_params_,
        'cv_scores': scores,
        f"mean_{scoring_method}": mean_scores,
        'p_value': p_value,
        'dropped_categories': rare_merger.dropped_categories_,
        'importances_df': importances_df
    }

    summary_data = {
        'best_params': str(result['best_params']),
        f"mean_{scoring_method}": result[f"mean_{scoring_method}"],
        'p_value': result['p_value'],
        'cv_scores': ', '.join([f"{s:.3f}" for s in result['cv_scores']]),
        'dropped_categories': str(result['dropped_categories'])
    }

    summary_df = pd.DataFrame(list(summary_data.items()), columns=["Metric", "Value"])
    summary_df.to_csv(os.path.join(save_path, f"{save_prefix}_random_forest_summary_{scoring_method}.csv"), index=False)
    result['importances_df'].to_csv(os.path.join(save_path, f"{save_prefix}_random_forest_importances_{scoring_method}.csv"), index=False)

    return result


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

        result[f"N_{group_col_name}={group2_name}"] = len(group2)
        result[f"Mean_{group_col_name}={group2_name}"] = group2.mean()

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


def chi_squared_test(contingency_table, include_expected=False):
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    result_df = pd.DataFrame({
        "test": ["chi squared"],
        "statistic": [chi2],
        "p": [p],
        "df": [dof],  # Degrees of Freedom
    })
    if include_expected:
        expected_df = pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns)
        return result_df, expected_df
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
    df_pivot = df_pivot.copy()
    df_pivot.loc[:, "Cluster"] = kmeans.fit_predict(df_pivot)

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

    return df_pivot, kmeans, cluster_centroids


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



