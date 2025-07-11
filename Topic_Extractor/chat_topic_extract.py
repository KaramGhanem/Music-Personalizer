import os
import re
import json
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass

import nltk
import spacy
import phrasemachine
import umap.umap_ as umap

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from joblib import Parallel, delayed


nlp = spacy.load("en_core_web_lg")
nltk.download("punkt")
nltk.download("stopwords")

LLM = ChatOpenAI(
    model='gpt-4o-mini',
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class ClusterSummary:
    response: str
    strategy: str

def is_alphabetic(phrase: str) -> bool:
    return re.fullmatch(r'[A-Za-z\s]+', phrase) is not None

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def vectorize_phrase(phrases: List[str]) -> np.ndarray:
    return np.stack([nlp(p).vector for p in phrases])

def intra_cluster_variance(embeddings) -> float:
    centroid = np.mean(embeddings, axis=0)
    return np.mean(np.linalg.norm(embeddings - centroid, axis=1) ** 2)

def reduce_embeddings(embeddings, n_neighbors=30, min_dist=0.02, n_components=5, metric='cosine'):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=42
    )
    return reducer.fit_transform(embeddings)

def determine_optimal_k(embeddings) -> int:
    scores = []
    for k in range(2, min(10, len(embeddings))):
        labels = KMeans(n_clusters=k, n_init=10).fit_predict(embeddings)
        scores.append(silhouette_score(embeddings, labels))
    return np.argmax(scores) + 2

def evaluate_config(n_components, n_neighbors, min_dist, metric, embeddings, freqs, lambda_weight=0.6):
    try:
        reduced = reduce_embeddings(embeddings, n_neighbors, min_dist, n_components, metric)
        k = determine_optimal_k(reduced)
        labels = KMeans(n_clusters=k, random_state=42).fit_predict(reduced, sample_weight=freqs)
        sil = silhouette_score(reduced, labels)
        variance = np.mean([intra_cluster_variance(reduced[labels == i]) for i in set(labels)])
        score = sil - lambda_weight * variance
        return {
            "score": score,
            "params": dict(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, k=k),
            "reduced": reduced,
            "silhouette": sil,
            "variance": variance
        }
    except Exception as e:
        logging.warning(f"Failed config: {e}")
        return None

class TopicSummarizer:
    def __init__(self, llm=None):
        self.llm = llm or LLM
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            ("human", "User Input: {phrases}")
        ])
        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def summarize(self, system_prompt: str, phrases: str) -> ClusterSummary:
        try:
            output = self.chain.invoke({
                "system_prompt": system_prompt,
                "phrases": phrases
            })
            return ClusterSummary(output, "summary")
        except Exception as e:
            logging.error(f"LLM error: {e}")
            return ClusterSummary(str(e), "error")

def extract_topics(chat_history: List[Dict[str, str]]) -> Dict[int, Dict[str, str]]:
    phrase_counts = defaultdict(int)
    for msg in chat_history:
        role = msg["role"]
        doc = nlp(msg["content"])
        tokens = [t.text for t in doc]
        pos = [t.pos_ for t in doc]
        phrases = phrasemachine.get_phrases(tokens=tokens, postags=pos)["counts"]
        for phrase, count in phrases.items():
            if is_alphabetic(phrase):
                phrase_counts[phrase] += count if role == "user" else count // 2

    threshold = np.percentile(list(phrase_counts.values()), 85)
    filtered = {p: c for p, c in phrase_counts.items() if c >= threshold}
    phrases, freqs = list(filtered.keys()), list(filtered.values())

    embeddings = vectorize_phrase(phrases)

    param_grid = [
        (nc, nn, md, m)
        for nc in range(2, 6)
        for nn in [20, 30]
        for md in [0.01, 0.02]
        for m in ["cosine", "euclidean"]
    ]

    results = Parallel(n_jobs=8)(
        delayed(evaluate_config)(nc, nn, md, m, embeddings, freqs)
        for (nc, nn, md, m) in param_grid
    )
    results = [r for r in results if r]
    best = max(results, key=lambda x: x["score"])

    kmeans = KMeans(n_clusters=best["params"]["k"], random_state=42)
    labels = kmeans.fit_predict(best["reduced"], sample_weight=freqs)

    clusters = defaultdict(list)
    for i, lbl in enumerate(labels):
        clusters[lbl].append((phrases[i], freqs[i]))

    summarizer = TopicSummarizer()
    summaries = {}

    for cid, items in clusters.items():
        phrase_list = ", ".join(p for p, _ in items)
        freq_list = ", ".join(str(f) for _, f in items)
        prompt = (
            "You are an assistant summarizing topics from chat logs. "
            "Return a JSON with 'Topic Title' and 'Topic Description' under 50 words. "
            "Do not use code fences. Output only a valid JSON object with these two keys."
        )
        response = summarizer.summarize(prompt, f"Phrases: {phrase_list}. Frequencies: {freq_list}")
        try:
            parsed = json.loads(response.response.strip())
            summaries[cid] = {
                "Topic Title": parsed.get("Topic Title", "N/A"),
                "Topic Description": parsed.get("Topic Description", "N/A")
            }
        except Exception:
            summaries[cid] = {
                "Topic Title": "Error",
                "Topic Description": response.response
            }

    return summaries
