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

# Initialize the language model (using ChatOpenAI as an example)
LLM = ChatOpenAI(
            model='gpt-4o-mini', 
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )

# Initialize logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

nlp = spacy.load("en_core_web_lg")
nltk.download("punkt")
nltk.download("stopwords") 

@dataclass
class ClusterTitleDescription:
    """
    Structured response format.
    """
    response: str
    strategy: str

def determine_optimal_clusters(embeddings) -> int:
    """
    Determines the optimal number of clusters based on dataset size using 
    the Elbow Method and Silhouette Score.
    
    :param embeddings: List or numpy array of phrase embeddings
    :return: Optimal number of clusters
    """
    n_samples = len(embeddings)
    
    # Start with a divisor of 4 and reduce until we get at least 3 clusters.
    divisor = 4
    while divisor > 1 and int(n_samples / divisor) < 4:
        divisor -= 1

    # Use the adjusted divisor to compute max clusters.
    max_clusters = int(n_samples / divisor)
    if max_clusters < 4:
        max_clusters = 4  # Ensure at least up to 4 clusters are tested

    possible_clusters = range(2, max_clusters)
    silhouette_scores = []
    
    for k in possible_clusters:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        silhouette_scores.append(silhouette_score(embeddings, labels))

    # Find the best Silhouette Score
    optimal_k_silhouette = possible_clusters[np.argmax(silhouette_scores)]
    optimal_k = optimal_k_silhouette  # You can set a bias if needed

    logging.info(f"Number of extracted phrases: {n_samples}")
    logging.info(f"Optimal k (Silhouette Score): {optimal_k_silhouette}")

    return optimal_k


def get_phrase_vector(phrases: List[str]) -> Tuple[np.ndarray, List[float]]:
    """
    Takes in phrase tuple list and returns a list of 300D phrase vectors 
    """
    phrases_vectors = []
    phrases_vector_norm = []
    for p in phrases:
        p_tokens = nlp(p)
        phrases_vectors.append(p_tokens.vector)
        phrases_vector_norm.append(p_tokens.vector_norm)
    phrases_vectors = np.stack(phrases_vectors, axis=0)
    return phrases_vectors, phrases_vector_norm 

def reduce_dimensions(embeddings, n_neighbors=30, min_dist=0.02, n_components=5, metric='cosine'):
    """Reduce embedding dimensionality using UMAP."""
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, 
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=42
    )
    return reducer.fit_transform(embeddings)

def is_alphabetic_phrase(phrase: str) -> bool:
    """
    Returns True if the phrase contains only alphabetic characters and spaces.
    """
    return re.fullmatch(r'[A-Za-z\s]+', phrase) is not None

def intra_cluster_variance(embeddings) -> float:
        centroid = np.mean(embeddings, axis=0)
        variances = np.linalg.norm(embeddings - centroid, axis=1) ** 2
        return np.mean(variances)

def evaluate_params(n_components: int, n_neighbors: int, min_dist: float, metric: str, embeddings, frequencies: List[float], lambda_weight: float = 0.6) -> Optional[Dict[str, Any]]:
    # Reduce dimensions using UMAP with given hyperparameters
    reduced_emb = reduce_dimensions(embeddings, n_neighbors, min_dist, n_components, metric)
    try:
        optimal_k = determine_optimal_clusters(reduced_emb)
    except Exception as e:
        logging.error(f"Error for params ({n_components}, {n_neighbors}, {min_dist}, {metric}): {e}")
        return None

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(reduced_emb, sample_weight=frequencies)
    
    # Compute silhouette score
    try:
        sil_score = silhouette_score(reduced_emb, labels)
    except Exception:
        sil_score = -1
    
    # Organize indices by cluster and compute average intra-cluster variance
    clusters_dict = {}
    cluster_variances = []

    for idx, label in enumerate(labels):
        # Organize indices by cluster
        clusters_dict.setdefault(label, []).append(idx)
        
        # Compute variance 
        if len(clusters_dict[label]) > 1:
            cluster_emb = reduced_emb[clusters_dict[label]]
            variance = intra_cluster_variance(cluster_emb)
            cluster_variances.append(variance)
        else:
            cluster_variances.append(0)

    avg_variance = np.mean(cluster_variances)
    
    # Compute a combined score
    combined_score = sil_score - lambda_weight * avg_variance

    return {
        "score": combined_score,
        "silhouette": sil_score,
        "variance": avg_variance,
        "params": {
            'n_components': n_components,
            'n_neighbors': n_neighbors,
            'min_dist': min_dist,
            'metric': metric,
            'optimal_k': optimal_k
        },
        "reduced_emb": reduced_emb
    }

class ClusterDescription:
    def __init__(self, llm: Optional[Any] = None, logging_enabled: bool = False) -> None:
        # Initialize the language model (using ChatOpenAI as an example)
        self.llm = llm or LLM
        
        # Optionally set up logging
        self.logger = self._setup_logging() if logging_enabled else None
        
        # Create a dynamic prompt template with a system message and a user input message.
        self.dynamic_prompt_template = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),  # Dynamic System Message
            ("human", "User Input: {extracted_phrases}")
        ])
        
        # Create a dynamic chain that pipes the prompt template into the LLM and then parses the output.
        self.dynamic_chain = (
            self.dynamic_prompt_template
            | self.llm
            | StrOutputParser()
        )

    def _setup_logging(self) -> Any:
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        return logger

    def run(self, system_prompt: str, extracted_phrases: str) -> ClusterTitleDescription:
        # Execute the chain with the provided prompt values.
        try:
            # Prepare input for the chain
            chain_input = {
                "system_prompt": system_prompt,
                "extracted_phrases": extracted_phrases
            }
            
            # Invoke the dynamic prompt template
            response = self.dynamic_chain.invoke(chain_input)

            return ClusterTitleDescription(
                response=response,
                strategy="cluster title and description",
            )

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error generating direct response: {e}")

            return ClusterTitleDescription(
                response=f"I encountered an error while generating a cluster title: {e}",
                strategy="error",
            )

def merge_conversations(chat_history: List[Dict[str, str]], max_chunk_size: Optional[int] = None) -> List[str]:
    # If max_chunk_size is not specified, merge all entries into one string.
    if max_chunk_size is None:
        return [" ".join(entry["content"] for entry in chat_history)]
    
    merged_documents = []
    temp_chunk = []
    for entry in chat_history:
        temp_chunk.append(entry["content"])  # or use preprocess_text(entry["content"]) if needed
        if len(temp_chunk) >= max_chunk_size:
            merged_documents.append(" ".join(temp_chunk))
            temp_chunk = []
    if temp_chunk:
        merged_documents.append(" ".join(temp_chunk))
    return merged_documents

def generate_topic_description(
    phrases: List[str], 
    highlighted_user_query: List[Dict[str, Any]]
) -> ClusterTitleDescription:
        """
        Uses the LLMParser to generate a topic description given a list of phrases and each phrase's representative sentence.
        The prompt concatenates the phrases and their corresponding representative sentence and asks the LLM for a concise description.
        """
        # Instantiate the parser
        parser = ClusterDescription()
        
        # Define the system prompt to instruct the LLM on its task.
        system_prompt = (
            "You are an assistant that generates topic descriptions based on a number of input key phrases "
            "and based on a highlighted piece of the chat history that these phrases were found in. These input phrases have been extracted from "
            "a chat history between a user and an agent. "
            "Considering the mentioned context, provide a concise (less than 8 words) generalized description of the overarching topic they represent in the form of a topic title. "
            "In addition please provide a more detailed description of the topic up to 50 words. "
            "A frequency will be provided for each phrase. This frequency represents a weight that should influence how strongly the generated topic title and topic description "
            "reflect the prominence of the corresponding text from which the phrase was extracted. This weight indicates how prominent or important that phrase is in the conversation. "
            "When generating the topic title and description, the model should give more influence to phrases with higher frequencies, ensuring that the final output reflects the most significant content from the original text. "
            "ALWAYS output the extracted topic title and the Topic description as a JSON as follows."
            "You are an assistant that must always output valid JSON with the keys:\n"
            "  \"Topic Title\" and \"Topic Description\"\n"
            "No code fences, no extra keys, only these keys in a top-level JSON object"
            "{\n"
            "  \"Topic Title\": \"Your short title here\",\n"
            "  \"Topic Description\": \"Your detailed description here\"\n"
            "}\n\n"
        )
        
        # Prepare the user query by concatenating the phrases.
        extracted_phrases = (
            f"Given the following extracted phrases from the user query: {', '.join(phrases)},\n\n"
            f"Given the following phrases' highlighted piece of the chat history that these phrases were found in : {', '.join([item['representative_sentence'] for item in highlighted_user_query])},\n\n"
            f"and given the following corresponding frequency values of each phrase : {', '.join([str(item['frequency']) for item in highlighted_user_query])}"
        )
        
        # Run the dynamic chain to generate the description.
        return parser.run(system_prompt=system_prompt, extracted_phrases=extracted_phrases)
    
def extract_topic_details(description: ClusterTitleDescription) -> Tuple[Optional[str], Optional[str]]:
    try:
        # Remove code block markers if present
        response = description.response.strip()
        response = re.sub(r'^```json\s*', '', response)
        response = re.sub(r'\s*```$', '', response)    
        # Now parse the cleaned JSON string
        parsed = json.loads(response)
        topic_title = parsed.get("Topic Title", "").strip()
        topic_description = parsed.get("Topic Description", "").strip()
        return topic_title, topic_description
    except Exception as e:
        logging.error(f"Error parsing topic details: {e}")
        return None, None

def cosine_similarity(vec1, vec2) -> float: 
    """Compute cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_sentence_embedding(sentence: str):
    """
    Given a sentence (string), return its vector embedding using spaCy.
    """
    doc = nlp(sentence)
    return doc.vector

def get_sentences_with_phrase(phrase: str, sentences: List[Any]) -> List[str]:
    """
    Given a phrase and a list of spaCy sentence spans,
    return the list of sentences that contain the phrase,
    using regex for a whole-word match.
    """
    matching_sentences = []
    phrase_lower = phrase.lower()
    # Build a regex pattern with word boundaries
    pattern = r'\b' + re.escape(phrase_lower) + r'\b'
    
    for sent in sentences:
        # Perform a case-insensitive search using regex
        if re.search(pattern, sent.text.lower()):
            matching_sentences.append(sent.text.strip())
    return matching_sentences

def representative_sentence_for_phrase(phrase: str, sentences: List[Any]) -> Optional[str]:
    """
    For a given phrase, find all matching sentences, compute the average sentence embedding,
    and return the sentence whose embedding is closest (by cosine similarity) to that average.
    """
    matching_sentences = get_sentences_with_phrase(phrase, sentences)
    if not matching_sentences:
        return None  # or return an empty string if no sentence matches
    # Compute embeddings for each matching sentence
    sent_embeddings = np.array([get_sentence_embedding(sent) for sent in matching_sentences])
    # Compute the centroid (average embedding) of these sentences
    centroid_embedding = np.mean(sent_embeddings, axis=0)
    # Compute cosine similarity of each sentence embedding to the centroid
    similarities = [cosine_similarity(vec, centroid_embedding) for vec in sent_embeddings]
    # Choose the sentence with the highest similarity
    rep_index = np.argmax(similarities)
    return matching_sentences[rep_index]

def extract_topics(chat_history: Optional[List[Dict]])-> Optional[Dict[int, Dict[str, str]]]:
    """
    Processes a given chat history to identify key topics discussed in the conversation. 
    It follows a multi-step process involving phrase extraction, frequency analysis, clustering, and topic summarization using an LLM. 
    The function returns structured topic descriptions based on clustered key phrases from the chat.

    :param embeddings: Chat Histiroy
    :return: Extracted Topics, Extracted phrases with frequencies before processing, Phrases after processing with frequencies and representative sentences
    """

    # -------- Step 1: Phrase Extraction --------

    # Initialize a dictionary to track phrase counts by role.
    # Each phrase will map to a dictionary with keys "user" and "assistant".
    phrase_roles = defaultdict(lambda: {"user": 0, "assistant": 0})

    for entry in chat_history:
        role = entry["role"].lower()  # e.g., "user" or "assistant"
        text = entry["content"]
        
        # Tokenize and tag the text for this message
        doc = nlp(text)
        tokens = [token.text for token in doc]
        pos_tags = [token.pos_ for token in doc]
        
        # Extract phrases using phrasemachine
        phrases = phrasemachine.get_phrases(tokens=tokens, postags=pos_tags, minlen=1)["counts"]
        
        # Filter out non-alphabetic phrases
        for phrase, count in phrases.items():
            if is_alphabetic_phrase(phrase):
                phrase_roles[phrase][role] += count

    meta_data_phrase_roles = phrase_roles #used to output phrases with their frequencies before processing as metadata
    for phrase, counts in phrase_roles.items():
        # Calculate adjusted assistant count
        if counts["assistant"] > 1:
            x = int(counts["assistant"] / 2)
        else:
            x = counts["assistant"]  # This handles 0 or 1
        
        # Update total frequency in both dictionaries
        counts["frequency"] = counts["user"] + x
        meta_data_phrase_roles[phrase]["frequency"] = counts["user"] + counts["assistant"]

    # Convert dictionary to a list of tuples if needed
    phrase_freq_pm = list(phrase_roles.items())

    # Log the results
    for phrase, counts in phrase_freq_pm:
        logging.info(f"{phrase}: {counts}")

    # ----- Dynamic Frequency Threshold -----
    freq_all_values = np.array([freq for _, freq in phrase_freq_pm])
    freq_values = [d['frequency'] for d in freq_all_values]
    # Define a dynamic threshold, for example, using the 85th percentile
    dynamic_threshold = np.percentile(freq_values, 85)
    logging.info("Dynamic threshold (85th percentile): %s", dynamic_threshold)

    # Filter out phrases that have a frequency lower than the dynamic threshold
    filtered_phrase_freq = [(phrase, freq) for phrase, freq in phrase_freq_pm if freq['frequency'] > dynamic_threshold]

    # For every longer phrase, check if any shorter phrase (word-boundary match) is contained in it.
    adjusted_freq = {phrase: freq['frequency'] for phrase, freq in filtered_phrase_freq}
    sorted_phrases = sorted(adjusted_freq.items(), key=lambda x: len(x[0]), reverse=True)
    for phrase_long, freq_long in sorted_phrases:
        for phrase_short in list(adjusted_freq.keys()):
            if phrase_short == phrase_long:
                continue
            # Use regex to check whole word occurrence:
            # This will match if phrase_short appears as a separate word in phrase_long.
            if re.search(r'\b' + re.escape(phrase_short) + r'\b', phrase_long):
                # Subtract the frequency of the longer phrase from the shorter phrase.
                adjusted_freq[phrase_short] = max(0, adjusted_freq[phrase_short] - freq_long)

    # Sanity check to remove phrases whose adjusted frequency becomes less than threshold.
    adjusted_freq = {phrase: freq for phrase, freq in adjusted_freq.items() if freq > dynamic_threshold}

    # Sort alphabetically by phrase
    phrase_freq_pm = sorted(adjusted_freq.items(), key=lambda x: x[0])

    logging.info("Extracted phrases and their frequencies:")
    for pf in phrase_freq_pm:
        logging.info(pf)

    # -------- Step 2: Phrase Clustering --------

    extracted_phrases = [pf[0] for pf in phrase_freq_pm]
    frequencies = [freq for _, freq in phrase_freq_pm]

    # get phrase embeddings
    embeddings, normalized_embeddings = get_phrase_vector(extracted_phrases)

    #####################################################

    # Define hyperparameter ranges for grid search of UMAP
    n_components_range = range(2, min(10, embeddings.shape[0] - 4))   
    n_neighbors_range = [20, 25, 30, 35]     
    min_dist_range = [0.01, 0.02, 0.03] 
    metrics = ['cosine', 'euclidean']

    # Create a list of all parameter combinations
    param_combinations = [
        (n_components, n_neighbors, min_dist, metric)
        for n_components in n_components_range
        for n_neighbors in n_neighbors_range
        for min_dist in min_dist_range
        for metric in metrics
    ]

    # Run grid search in parallel multiprocessing (default parameter for Parallel is backend="loky"/ can switch to threading)
    results = Parallel(n_jobs=8, verbose=10)(
        delayed(evaluate_params)(n_components, n_neighbors, min_dist, metric, embeddings, frequencies)
        for (n_components, n_neighbors, min_dist, metric) in param_combinations
    )

    results = [res for res in results if res is not None]

    # Select best configuration based on the combined score (higher is better)
    best_result = max(results, key=lambda x: x["score"])
    best_params = best_result["params"]
    best_reduced_embeddings = best_result["reduced_emb"]

    logging.info("\nBest UMAP parameters based on combined metric:")
    logging.info(best_params)
    logging.info(f"Silhouette Score: {best_result['silhouette']:.3f}, Average Variance: {best_result['variance']:.6f}, Combined Score: {best_result['score']:.3f}")

    # Once best parameters are determined, perform final clustering using sample weights
    final_kmeans = KMeans(n_clusters=best_params['optimal_k'], random_state=42, n_init=10)
    final_labels = final_kmeans.fit_predict(best_reduced_embeddings, sample_weight=frequencies)

    # Group phrases by cluster, including their frequency
    clustered_phrases = {}
    for idx, label in enumerate(final_labels):
        clustered_phrases.setdefault(label, []).append({
            "phrase": extracted_phrases[idx],
            "frequency": frequencies[idx]
        })

    # Output the final clusters
    logging.info("\nFinal Clusters:")
    for cluster_id, items in clustered_phrases.items():
        logging.info(f"Cluster {cluster_id}:")
        for item in items:
            logging.info(f"  - {item['phrase']} (freq: {item['frequency']})")


    # Split the text into sentences for matching (using sentencizer)
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    doc = nlp(merge_conversations(chat_history)[0])
    sentences = list(doc.sents)

    # For each phrase in each cluster, determine a representative sentence.
    cluster_phrase_rep_sentences = {}  # Structure: {cluster_id: {phrase: representative_sentence, frequency}, ...}

    for cluster_id, phrase_list in clustered_phrases.items():
        cluster_phrase_rep_sentences[cluster_id] = {}
        for item in phrase_list:
            phrase = item['phrase']
            frequency = item['frequency']
            rep_sentence = representative_sentence_for_phrase(phrase, sentences)
            if rep_sentence is not None:
                cluster_phrase_rep_sentences[cluster_id][phrase] = {
                    "representative_sentence": rep_sentence,
                    "frequency": frequency
                }
            else:
                cluster_phrase_rep_sentences[cluster_id][phrase] = {
                    "representative_sentence": "(No matching sentence found)",
                    "frequency": frequency
                }

    # Log the representative sentences for each phrase in each cluster:
    logging.info("Representative sentences for each phrase in each cluster:")
    for cluster_id, phrases_dict in cluster_phrase_rep_sentences.items():
        logging.info(f"Cluster {cluster_id}:")
        for phrase, rep_sent in phrases_dict.items():
            logging.info(f"{phrase} --> {rep_sent}")
        logging.info("")

    # -------- TOPICS GENERATED WITH EXTRACTED PHRASES AND AVERAGE PHRASE SENTENCE -------- 

    # Initialize an empty dictionary to store topic descriptions
    cluster_title_descriptions = {}

    for cluster_id, sentences_info in cluster_phrase_rep_sentences.items():
        # Generate a description using the matched sentences
        description = generate_topic_description(list(sentences_info.keys()), list(sentences_info.values()))
        
        # Extract topic title and description
        topic_title, topic_description = extract_topic_details(description)
        
        # Store both title and description directly in the dictionary
        cluster_title_descriptions[cluster_id] = {
            "Topic Title": topic_title if topic_title else "N/A",
            "Topic Description": topic_description if topic_description else "N/A"
        }
            
    logging.info("=" * 150)
    # Log the results
    for cluster_id, topic_info in cluster_title_descriptions.items():
        logging.info("=" * 50)
        logging.info(f"Cluster number: {cluster_id}")
        logging.info(f"Topic Title: {topic_info['Topic Title']}")
        logging.info(f"Topic Description: {topic_info['Topic Description']}")
    
    return cluster_title_descriptions, meta_data_phrase_roles, cluster_phrase_rep_sentences