import os
import sys
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
        min_dist_range = [0.05, 0.1, 0.3, 0.5]
    if n_components_range is None:
        n_components_range = [2, 5, 10]
    if n_neighbors_range is None:
        n_neighbors_range = [2, 5, 15, 30]
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
                     metric='cosine')

        # Fit the UMAP model and compute the metric (Silhouette score here)
        embeddings_ = model.fit_transform(embeddings)

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
        min_samples_range = [2, 3, 5, 7, 10]
    if min_cluster_size_range is None:
        min_cluster_size_range = [3, 5, 10, 15, 20]
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

    # We need to ignore noise points labeled as -1 (which are considered outliers by HDBSCAN)
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


def train_bertopic(docs):
    embedding_model = SentenceTransformer("all-mpnet-base-v2")

    # Generate embeddings for the documents
    embeddings = generate_embeddings(docs, embedding_model)

    # Apply GridSearch on UMAP and HDBSCAN
    best_umap_model = optimize_umap(embeddings)
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


def main():
    import os
    FOLDER_PATH = r"C:\Users\Rony\Documents\projects\ethics\survey_analysis\data\analysis_data\all\exploratory\i_c_s"
    FILE_PATH = os.path.join(FOLDER_PATH, "goals_intentions without consciousness.csv")
    TEXT_COL = "Do you have an example of a case of goals/intentions without consciousness?"
    EXCLUDE_WORDS = ["goal", "goals", "intention", "intentions", "consciousness"]
    OUTPUT_DIR = r"C:\Users\Rony\Documents\projects\ethics\survey_analysis\data\analysis_data\all\exploratory\i_c_s\topic_modelling"
    # redirect all "prints" to a log file
    sys.stdout = open(os.path.join(OUTPUT_DIR, "output_log.txt"), "w")

    df = pd.read_csv(FILE_PATH)
    df = tokenize_responses(df=df, text_col=TEXT_COL, custom_stopwords=EXCLUDE_WORDS)

    topic_model, topics, probs = train_bertopic(df["cleaned"].tolist())
    cv_score, umass_score = evaluate_model(topic_model, df["cleaned"].tolist())
    labels = generate_llm_labels(topic_model)

    save_outputs(df=df, topic_model=topic_model, topics=topics, probs=probs, cv_score=cv_score, umass_score=umass_score,
                 labels=labels, output_dir=OUTPUT_DIR)
    print(f"Outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
