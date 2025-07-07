import pandas as pd
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import logging
import json
import ast
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()

# Replace with your actual values
GPT4_API_KEY = os.getenv("GPT4_API_KEY")
GPT4_ENDPOINT = os.getenv("GPT4_ENDPOINT")
GPT4_EMBEDDING_DEPLOYMENT_NAME = os.getenv("GPT4_EMBEDDING_DEPLOYMENT_NAME")
GPT4_EMBEDDING_API_VERSION = os.getenv("GPT4_EMBEDDING_API_VERSION")

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=GPT4_ENDPOINT,
    api_key=GPT4_API_KEY,
    api_version=GPT4_EMBEDDING_API_VERSION,
)

def get_embedding(text):
    """
    Generate an embedding for the given text using the Azure OpenAI text-embedding model.
    
    :param text: The input text to embed.
    :return: A list representing the embedding vector.
    """
    try:
        response = client.embeddings.create(
            input=text,
            model=GPT4_EMBEDDING_DEPLOYMENT_NAME
        )
        return response.data[0].embedding  # Return embedding as a Python list
    except Exception as e:
        logging.error(f"Error generating embedding for '{text}': {str(e)}")
        return None

def convert_numpy(DATA_PATH, embedding_column):
    """
    Load the product data and convert the embedding column to numpy arrays.
    
    :param DATA_PATH: Path to the input CSV file.
    :return: Numpy array of product vectors and the DataFrame.
    """
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    # Parse embedding column from string to numpy array
    def parse_embedding(x):
        try:
            emb = ast.literal_eval(x) if isinstance(x, str) else x
            return np.array(emb).flatten()  # Flatten to 1D array
        except:
            return None

    df[f"{embedding_column}"] = df[f"{embedding_column}"].apply(parse_embedding)
    df = df[df[f"{embedding_column}"].notnull()]  # Drop rows with invalid embeddings

    # Convert to numpy array and validate shape
    product_vectors = np.array(df[f"{embedding_column}"].tolist())
    logging.info(f"Shape of product_vectors: {product_vectors.shape}")  # Debugging line
    assert len(product_vectors.shape) == 2, "product_vectors must have 2 dimensions (N, D)"

    return product_vectors, df

def preprocess_text(text):
    """
    Preprocess text by lowercasing, removing punctuation, and tokenizing.
    
    :param text: Input text.
    :return: List of tokens.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.split()

def bm25_search(query, corpus, top_k=5):
    """
    Perform BM25 search on the given corpus.
    
    :param query: User input query (string).
    :param corpus: List of documents (strings).
    :param top_k: Number of top results to return.
    :return: List of indices of top-k matching documents.
    """
    tokenized_corpus = [preprocess_text(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    # print("😂 bm25 ", bm25)

    tokenized_query = preprocess_text(query)
    scores = bm25.get_scores(tokenized_query)
    
    # print("😂 score ", scores)
    top_indices = np.argsort(scores)[::-1][:top_k]  # Get top-k indices
    # print("😂 top_indices ", top_indices)
    return top_indices

def hybrid_search(query, data_path, embedding_column, text_column, top_k=5, alpha=0.5):
    """
    Perform hybrid search using both vector search and BM25.
    
    :param query: User input query (string).
    :param df: DataFrame containing product data.
    :param embedding_column: Name of the embedding column.
    :param text_column: Name of the text column for BM25.
    :param top_k: Number of top results to return.
    :param alpha: Weight for vector search (BM25 weight is 1 - alpha).
    :return: DataFrame with top-k similar products.
    """
    # Vector Search
    query_vec = np.array(get_embedding(query)).flatten()
    if query_vec is None:
        logging.error("Failed to generate embedding for the query.")
        return None

    product_vectors, df = convert_numpy(data_path, embedding_column)
    vector_similarities = cosine_similarity([query_vec], product_vectors)[0]

    # BM25 Search
    corpus = df[text_column].tolist()
    bm25_indices = bm25_search(query, corpus, top_k=len(corpus))  # Get all indices
    bm25_similarities = np.zeros(len(corpus))
    bm25_similarities[bm25_indices] = np.linspace(1, 0, len(bm25_indices))  # Assign decreasing scores

    # Combine Scores
    combined_scores = alpha * vector_similarities + (1 - alpha) * bm25_similarities
    df["combined_score"] = combined_scores

    # Sort by combined score
    results = df.sort_values(by="combined_score", ascending=False).head(top_k)
    return results[["name", "price", "combined_score", "information_product", "url"]]

def test_bm25_serach(top_k):
    # Load product data
    DATA_PATH = "datasets/products_embedding_openai.csv"
    try:
        df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    except Exception as e:
        logging.error(f"Failed to load product data: {str(e)}")
        exit()

    # User input query
    user_input = "Tôi muốn mua Giày với mã SP 07"

    # BM25 search
    text_column = "name"
    corpus = df[text_column].tolist()
    bm25_indices = bm25_search(user_input, corpus, top_k=len(corpus)) 
    bm25_similarities = np.zeros(len(corpus))
    bm25_similarities[bm25_indices] = np.linspace(1, 0, len(bm25_indices)) 
    df["bm25_similarities"] = bm25_similarities
    results = df.sort_values(by="bm25_similarities", ascending=False).head(top_k)
    top_results = results[["name", "price", "bm25_similarities", "information_product", "url"]]

    if top_results is not None:
        for i, row in top_results.iterrows():
            print(f"👟 {row['name']} - Giá: {int(row['price']):,} VND - Combined Score: {row['bm25_similarities']:.4f}")
            print(f"📝 {row['information_product'][:100]}...")
            print(f"🔗 {row['url']}\n")

        # Save results to a temporary CSV file
        top_results.to_csv("hybrid_search_results.csv", index=False)


def test_hybrid_search(top_k):
    DATA_PATH = "datasets/products_embedding_openai.csv"

    # User input query
    user_input = "Tôi muốn mua Giày với mã  SP 07"

    # Perform hybrid search
    top_results = hybrid_search(
        query=user_input,
        data_path=DATA_PATH,
        embedding_column="information_product_embedding",
        text_column="information_product",
        top_k=top_k,
        alpha=0.7  # Weight for vector search (adjust as needed)
    )

    if top_results is not None:
        for i, row in top_results.iterrows():
            print(f"👟 {row['name']} - Giá: {int(row['price']):,} VND - Combined Score: {row['combined_score']:.4f}")
            print(f"📝 {row['information_product'][:100]}...")
            print(f"🔗 {row['url']}\n")

        # Save results to a temporary CSV file
        top_results.to_csv("hybrid_search_results.csv", index=False)


top_k = 10
test_bm25_serach(top_k)
# test_hybrid_search(top_k)