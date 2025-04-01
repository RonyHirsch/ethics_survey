import os
import sys
import random
import pickle
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score
from bertopic import BERTopic
from umap import UMAP
import hdbscan
from sentence_transformers import SentenceTransformer
from bertopic.vectorizers import ClassTfidfTransformer
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from tqdm import tqdm
import numpy as np
import re
import string
import warnings
import plotter


SEED = 17
random.seed(SEED)

warnings.filterwarnings("ignore")

if not os.path.exists(os.path.join("venv/nltk_data", "tokenizers/punkt")):
    nltk.download("punkt", download_dir="venv/nltk_data")
    if not os.path.exists(os.path.join("venv/nltk_data", "tokenizers/punkt_tab")):
        nltk.download("punkt_tab", download_dir="venv/nltk_data")
if not os.path.exists(os.path.join("venv/nltk_data", "corpora/stopwords")):
    nltk.download("stopwords", download_dir="venv/nltk_data")


def preprocess_text(text, custom_stopwords=None):
    if custom_stopwords is None:
        custom_stopwords = []
    text = text.lower()

    # Remove the phrase "I <something>" (e.g., "I think", "I know")
    for stopword in custom_stopwords:
        if stopword.lower().startswith("i "):  # if it begins with "I"
            text = text.replace(stopword.lower() + " ", "")  # delete it

    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    stop_words.update(custom_stopwords)
    filtered_tokens = [w for w in tokens if w not in stop_words]
    return " ".join(filtered_tokens)


def tokenize_responses(df, text_col, custom_stopwords=None):
    if custom_stopwords is None:
        custom_stopwords = []
    df["cleaned"] = df[text_col].apply(lambda x: preprocess_text(str(x), custom_stopwords))
    print(f"Total tokenized responses: {df.shape[0]}")
    return df


def generate_embeddings(docs, model):
    """Generate embeddings for a list of text documents using SentenceTransformer."""
    return model.encode(docs, show_progress_bar=True)


def optimize_umap(embeddings, n_neighbors_range=None, n_components_range=None, min_dist_range=None):
    if min_dist_range is None:
        min_dist_range = [0.05, 0.1, 0.2, 0.3, 0.5]
    if n_components_range is None:
        n_components_range = [2, 3, 4, 5, 10]
    if n_neighbors_range is None:
        n_neighbors_range = [2, 3, 4, 5, 10, 15, 30]
    best_umap_model = None
    best_score = -np.inf
    best_params = {}

    # Create parameter grid for UMAP
    param_grid = {
        'n_neighbors': n_neighbors_range,
        'n_components': n_components_range,
        'min_dist': min_dist_range,
    }

    # Search for best parameters with grid search
    for params in tqdm(ParameterGrid(param_grid), desc="UMAP Optimization"):
        model = UMAP(n_neighbors=params['n_neighbors'],
                     n_components=params['n_components'],
                     min_dist=params['min_dist'],
                     metric='cosine', random_state=SEED)

        # Fit the UMAP model and compute the silhouette score
        """
        Typically, embeddings have a large number of features (e.g., hundreds or thousands of dimensions) for each 
        sample. If the number of features (columns) is smaller than the number of samples (rows), or the data is sparse, 
        it can cause issues during dimensionality reduction.
        """
        try:
            embeddings_ = model.fit_transform(embeddings)
        except TypeError:
            if embeddings.shape[0] <= 1.5 * embeddings.shape[1]:
                print(f"Error: {embeddings.shape[0]} responses and {embeddings.shape[1]} features, this is too small")
                return best_umap_model
            else:
                raise Exception

        # Now optimize HDBSCAN based on these UMAP embeddings
        best_hdbscan_model = optimize_hdbscan(embeddings_)

        # Compute silhouette score after fitting HDBSCAN on the UMAP embeddings
        score = evaluate_umap_score(embeddings_, best_hdbscan_model, embeddings)

        # Update best model and parameters
        if score > best_score:
            best_score = score
            best_umap_model = model
            best_params = params

    print(f"Best UMAP params: {best_params}, score: {best_score}")

    return best_umap_model


def optimize_hdbscan(embeddings, min_cluster_size_range=None, min_samples_range=None):
    if min_samples_range is None:
        min_samples_range = [3, 4, 5, 7, 10]
    if min_cluster_size_range is None:
        min_cluster_size_range = [3, 4, 5, 7, 10, 15, 20]
    best_hdbscan_model = None
    best_score = -np.inf
    best_params = {}

    # Create parameter grid for HDBSCAN
    param_grid = {
        'min_cluster_size': min_cluster_size_range,
        'min_samples': min_samples_range,
    }

    # Search for best parameters with grid search
    for params in tqdm(ParameterGrid(param_grid), desc="HDBSCAN Optimization"):
        model = hdbscan.HDBSCAN(min_cluster_size=params["min_cluster_size"],
                                min_samples=params["min_samples"],
                                metric="euclidean")

        # Fit the HDBSCAN model
        model.fit(embeddings)

        # Evaluate the model using silhouette score
        score = evaluate_hdbscan_score(model, embeddings)

        # Update best model and parameters
        if score > best_score:
            best_score = score
            best_hdbscan_model = model
            best_params = params

    print(f"\nBest HDBSCAN params: {best_params}, score: {best_score}")

    return best_hdbscan_model


def evaluate_umap_score(embeddings, best_hdbscan_model, original_embeddings):
    # Get the cluster labels from HDBSCAN after fitting the model
    cluster_labels = best_hdbscan_model.labels_

    # If there is only one cluster, which would mean that all points are classified as noise (-1)
    if len(set(cluster_labels)) == 1 and -1 in set(cluster_labels):
        # Return a very low score indicating this is not a good result
        print("Only 'noise' (-1) cluster found")
        return -np.inf

    # Compute the silhouette score directly on the UMAP embeddings with cluster labels
    score = silhouette_score(original_embeddings, cluster_labels, metric='cosine')

    return score


def evaluate_hdbscan_score(model, embeddings):
    # Get the cluster labels from HDBSCAN
    cluster_labels = model.labels_

    # Check if there is more than one cluster
    if len(set(cluster_labels)) <= 1:  # Only one cluster or all points are noise (-1)
        print("Skipping silhouette score calculation. Only one cluster or all points are noise.")
        return -1  # Return a low score to indicate bad clustering

    # Compute silhouette score, excluding noise points labeled as -1
    score = silhouette_score(embeddings, cluster_labels, metric="euclidean")
    return score


def plot_umap_embeddings(embeddings, topics, labels, save_path, save_name, fmt="svg", dpi=1000, dynamic_label_size=True,
                         use_medoids=True, label_color_list=None):

    plotter.plotter_umap_embeddings(embeddings=embeddings, topics=topics, labels_dict=labels,
                                    save_path=save_path, save_name=save_name, fmt=fmt, dpi=dpi,
                                    dynamic_label_size=dynamic_label_size, use_medoids=use_medoids,
                                    label_color_list=label_color_list)
    return


def train_bertopic(docs, save_path):
    embedding_model = SentenceTransformer("all-mpnet-base-v2")

    # Generate embeddings for the documents
    embeddings = generate_embeddings(docs, embedding_model)

    # Apply GridSearch on UMAP and HDBSCAN
    best_umap_model = optimize_umap(embeddings)
    best_umap_params = best_umap_model.get_params()
    umap_embeddings = best_umap_model.transform(embeddings)

    # Now we set prediction_data=True explicitly for HDBSCAN
    best_hdbscan_model = optimize_hdbscan(umap_embeddings)

    # Make sure to generate prediction data for HDBSCAN
    best_hdbscan_model.set_params(prediction_data=True)  # Ensure prediction data is available

    vectorizer_model = CountVectorizer()
    ctfidf_model = ClassTfidfTransformer()

    # Initialize BERTopic with the best UMAP and HDBSCAN models
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=best_umap_model,
        hdbscan_model=best_hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        calculate_probabilities=True,
        verbose=True
    )

    # Fit the model
    topic_model.fit(docs)

    # Now transform the data to get the topics and probabilities
    topics, probs = topic_model.transform(docs)

    # Extract the UMAP embeddings from the trained model
    umap_embeddings = topic_model.umap_model.transform(embeddings)

    # labels
    topic_labels = generate_llm_labels(topic_model)


    """
    *** PLOT ***
    Our goal is to visualize in 2D while ensuring the most accurate representation of the data in two dimensions, 
    so we will set n_components=2 direcly. But we already have a UMAP model, which might have a higher number of
    components. So we will use the n_components=2 option when fitting the model. 
    """

    # list of colors for visualization:
    label_color_list = ['#1d3557', '#22333B', '#385f71', '#d7b377', '#C6AC8F', '#8f754f', '#5E503F',
                        '#5A6872', '#6C7A8A', '#7A8F99', '#9E9C8A', '#B5B59A', '#C8B5A3', '#D4C5A0',
                        '#6E4B3A', '#8F6E58']

    # Create a new UMAP model with n_components=2, but using the optimal parameters
    reduced_umap_model = UMAP(
        n_components=2,
        n_neighbors=best_umap_params["n_neighbors"],
        min_dist=best_umap_params["min_dist"],
        metric=best_umap_params["metric"],
        random_state=best_umap_params["random_state"]
    )
    reduced_umap_embeddings = reduced_umap_model.fit_transform(embeddings)
    plot_umap_embeddings(embeddings=reduced_umap_embeddings, topics=topics, labels=topic_labels,
                         save_path=save_path, save_name="umap_embeddings", fmt="svg", dpi=1000, dynamic_label_size=True,
                         use_medoids=True, label_color_list=label_color_list)

    return topic_model, topics, probs


def evaluate_model(topic_model, docs):
    docs_tokenized = [doc.split() for doc in docs]
    dictionary = Dictionary(docs_tokenized)
    corpus = [dictionary.doc2bow(text) for text in docs_tokenized]

    topic_words = []
    for topic_id in topic_model.get_topics():
        if topic_id == -1:
            continue
        words = [word for word, _ in topic_model.get_topic(topic_id)]
        topic_words.append(words)

    cm_cv = CoherenceModel(topics=topic_words, texts=docs_tokenized, dictionary=dictionary, coherence="c_v")
    cm_umass = CoherenceModel(topics=topic_words, texts=docs_tokenized, dictionary=dictionary, coherence="u_mass")
    return cm_cv.get_coherence(), cm_umass.get_coherence()


def generate_llm_labels(topic_model, top=10):
    topics = topic_model.get_topics()
    labels = {}
    for topic_id in topics:
        if topic_id == -1:
            continue
        keywords = [word for word, _ in topic_model.get_topic(topic_id)[:top]]  # top words
        labels[topic_id] = f"Label for: {keywords[0]} ({keywords[1: ]})"
    return labels


def save_outputs(df, topic_model, topics, probs, cv_score, umass_score, labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    df["topic"] = topics
    df["topic_label"] = df["topic"].apply(lambda topic: labels.get(topic, "Unknown").split(': ')[-1].split()[0] if topic != -1 else "Unknown")
    df["soft_topic_assignment"] = [probs[i][topics[i]] if topics[i] != -1 else 0.0 for i in range(len(topics))]
    df["all_topic_assignments"] = [probs[i] if topics[i] != -1 else [0.0] * len(probs[i]) for i in range(len(topics))]
    df.to_csv(os.path.join(output_dir, "preprocessed_responses.csv"), index=False)

    # save the BERTopic model
    with open(os.path.join(output_dir, "bertopic_model.pkl"), "wb") as f:
        pickle.dump(topic_model, f)

    # all responses grouped by topics
    topic_groups = df.groupby("topic").apply(lambda group: group["cleaned"].tolist())
    topic_groups_df = topic_groups.reset_index(name='responses')
    topic_groups_df.to_csv(os.path.join(output_dir, "responses_by_topic.csv"), index=False)

    # save the coherence scores
    """
    Coherence CV tends to correlate well with human judgment of topic quality.
    Coherence UMass is more of a statistical measure of topic coherence and is typically used in information retrieval.
    """
    with open(os.path.join(output_dir, "coherence_scores.txt"), "w") as f:
        f.write(f"Coherence CV: {cv_score:.4f}\n")
        f.write(f"Coherence UMass: {umass_score:.4f}\n")

    # save the topic labels
    with open(os.path.join(output_dir, "topic_assignments.csv"), "w") as f:
        for topic_id, label in labels.items():
            f.write(f"Topic {topic_id}: {label}\n")


def main(file_path, output_path, text_col, exclude_words):
    # redirect all "prints" to a log file
    sys.stdout = open(os.path.join(output_path, "output_log.txt"), "w")

    df = pd.read_csv(file_path)
    # filter out responses where the relevant column is empty
    df = df.dropna(axis=0, subset=[text_col]).reset_index(drop=True, inplace=False)
    df = tokenize_responses(df=df, text_col=text_col, custom_stopwords=exclude_words)
    print(f"Total number of responses: {df.shape[0]}")

    topic_model, topics, probs = train_bertopic(df["cleaned"].tolist(), save_path=output_path)
    cv_score, umass_score = evaluate_model(topic_model, df["cleaned"].tolist())
    labels = generate_llm_labels(topic_model)

    save_outputs(df=df, topic_model=topic_model, topics=topics, probs=probs, cv_score=cv_score, umass_score=umass_score,
                 labels=labels, output_dir=output_path)
    print(f"Outputs saved to {output_path}")
    print(f"list of excluded words: \n{exclude_words}")
    return


if __name__ == "__main__":

    """
    Kill for test - not killing any creature
    """
    TEXT_COL = "noKill_Other: please specify"
    FOLDER_PATH = r"C:\Users\Rony\Documents\projects\ethics\survey_analysis\data\analysis_data\all\exploratory\kill_for_test"
    FILE_PATH = os.path.join(FOLDER_PATH, "kill_to_pass.csv")
    OUTPUT_NAME = "all_no_other"
    OUTPUT_DIR = os.path.join(FOLDER_PATH, "topic_modelling", OUTPUT_NAME)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    EXCLUDE_WORDS = ["kill", "pass", "test"]
    main(file_path=FILE_PATH, output_path=OUTPUT_DIR, text_col=TEXT_COL, exclude_words=EXCLUDE_WORDS)
    exit()


    """
    Consciousness and intelligence - related to the same third factor
    """
    TEXT_COL = "What is the common denominator?"
    FOLDER_PATH = r"C:\Users\Rony\Documents\projects\ethics\survey_analysis\data\analysis_data\all\exploratory\consciousness_intelligence"
    FILE_PATH = os.path.join(FOLDER_PATH, "common_denominator.csv")
    OUTPUT_NAME = "common_denominator"
    OUTPUT_DIR = os.path.join(FOLDER_PATH, "topic_modelling", OUTPUT_NAME)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    EXCLUDE_WORDS = ["intelligent", "intelligence", "conscious", "consciousness", "related", "common"]
    main(file_path=FILE_PATH, output_path=OUTPUT_DIR, text_col=TEXT_COL, exclude_words=EXCLUDE_WORDS)


    """
    I_C_S
    """

    GLOBAL_EXCLUSION_LIST = ["without", "I think", "I would", "I believe", "I don't", "I dont", "Maybe"]

    """
    Concsciousness w/o intentions/goals
    """
    # question
    TEXT_COL = "Do you have an example of a case of consciousness without intentions/goals?"

    # data
    FOLDER_PATH = r"C:\Users\Rony\Documents\projects\ethics\survey_analysis\data\analysis_data\all\exploratory\i_c_s"
    FILE_PATH = os.path.join(FOLDER_PATH, "consciousness without intentions_goals.csv")
    OUTPUT_NAME = "consciousness_wo_intentions"
    OUTPUT_DIR = os.path.join(FOLDER_PATH, "topic_modelling", OUTPUT_NAME)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # global and specific
    EXCLUDE_WORDS = GLOBAL_EXCLUSION_LIST + ["intention", "intentions", "goal", "goals", "consciousness", "conscious",
                                             "concious"]

    main(file_path=FILE_PATH, output_path=OUTPUT_DIR, text_col=TEXT_COL, exclude_words=EXCLUDE_WORDS)



    """
    Concsciousness w/o sensations
    """
    # question
    TEXT_COL = "Do you have an example of a case of consciousness without sensations of pleasure or pain?"

    # data
    FOLDER_PATH = r"C:\Users\Rony\Documents\projects\ethics\survey_analysis\data\analysis_data\all\exploratory\i_c_s"
    FILE_PATH = os.path.join(FOLDER_PATH, "consciousness without sensations of pleasure or pain.csv")
    OUTPUT_NAME = "consciousness_wo_sensations"
    OUTPUT_DIR = os.path.join(FOLDER_PATH, "topic_modelling", OUTPUT_NAME)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    EXCLUDE_WORDS = GLOBAL_EXCLUSION_LIST + ["pain", "pleasure", "sensation", "sensations",
                                             "consciousness", "conscious", "consciously"]
    main(file_path=FILE_PATH, output_path=OUTPUT_DIR, text_col=TEXT_COL, exclude_words=EXCLUDE_WORDS)



    """
    Intentions/Goals without consciousness
    """
    # question
    TEXT_COL = "Do you have an example of a case of goals/intentions without consciousness?"

    # data
    FOLDER_PATH = r"C:\Users\Rony\Documents\projects\ethics\survey_analysis\data\analysis_data\all\exploratory\i_c_s"
    FILE_PATH = os.path.join(FOLDER_PATH, "goals_intentions without consciousness.csv")
    OUTPUT_NAME = "intentions_wo_consciousness"
    OUTPUT_DIR = os.path.join(FOLDER_PATH, "topic_modelling", OUTPUT_NAME)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    EXCLUDE_WORDS = GLOBAL_EXCLUSION_LIST + ["goal", "goals", "intention", "intentions", "consciousness", "conscious"]
    main(file_path=FILE_PATH, output_path=OUTPUT_DIR, text_col=TEXT_COL, exclude_words=EXCLUDE_WORDS)


    """
    Sensations w/o consciousness
    """
    # question
    TEXT_COL = "Do you have an example of a case of positive/negative sensations without consciousness?"

    # data
    FOLDER_PATH = r"C:\Users\Rony\Documents\projects\ethics\survey_analysis\data\analysis_data\all\exploratory\i_c_s"
    FILE_PATH = os.path.join(FOLDER_PATH, "positive_negative sensations without consciousness.csv")
    OUTPUT_NAME = "sensations_wo_consciousness"
    OUTPUT_DIR = os.path.join(FOLDER_PATH, "topic_modelling", OUTPUT_NAME)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    EXCLUDE_WORDS = GLOBAL_EXCLUSION_LIST + ["positive", "negative", "sensation", "sensations", "consciousness"]
    main(file_path=FILE_PATH, output_path=OUTPUT_DIR, text_col=TEXT_COL, exclude_words=EXCLUDE_WORDS)

















