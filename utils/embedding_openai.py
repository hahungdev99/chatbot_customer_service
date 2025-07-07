import pandas as pd
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import logging
import ssl
import ast
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


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
        # Ensure the embedding is returned as a Python list
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding for '{text}': {str(e)}")
        return None

def embed_data(input_file_path, output_file_path, column_name):
    """
    Embed data from a specified column in a CSV file and save the results to a new file.
    
    :param input_file_path: Path to the input CSV file.
    :param output_file_path: Path to save the output CSV file with embeddings.
    :param column_name: Name of the column to embed.
    """
    # Load the input CSV file
    print(f"📂 Loading data from: {input_file_path}")
    df = pd.read_csv(input_file_path)
    
    # Check if the specified column exists
    if column_name not in df.columns:
        print(f"Error: Column '{column_name}' not found in the input CSV file.")
        return
    
    # Encode the specified column into embeddings
    print(f"Generating embeddings for column: {column_name}")
    df[f"{column_name}_embedding"] = df[column_name].apply(
        lambda x: [get_embedding(x)] if isinstance(x, str) else None
    )
    
    # Drop rows where embedding generation failed
    df = df[df[f"{column_name}_embedding"].notnull()]
    
    # Save the updated DataFrame to the output CSV file
    print(f"Saving embedded data to: {output_file_path}")
    df.to_csv(output_file_path, index=False, encoding="utf-8-sig")
    print("Embedding process completed successfully.")

def generate_single_embedding(input_text):
    """
    Generate an embedding for a single input text and print it.
    
    :param input_text: The input text to embed.
    """
    embedding = get_embedding(input_text)
    if embedding:
        print(f"Input Text: {input_text}")
        print(f"Embedding Vector (length={len(embedding)}):")
        print(embedding[:10], "...")  # Print the first 10 dimensions for brevity
    else:
        print(f"Failed to generate embedding for: {input_text}")

# Convert CSV file's embedding column from string to numpy array
def convert_numpy(DATA_PATH):
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

    df["name_embedding"] = df["name_embedding"].apply(parse_embedding)
    df = df[df["name_embedding"].notnull()]  # Drop rows with invalid embeddings

    # Convert to numpy array and validate shape
    product_vectors = np.array(df["name_embedding"].tolist())
    logging.info(f"Shape of product_vectors: {product_vectors.shape}")  # Debugging line
    assert len(product_vectors.shape) == 2, "product_vectors must have 2 dimensions (N, D)"

    return product_vectors, df

# Search for similar products
def search_similar_products(query: str, top_k=5):
    """
    Search for products similar to the query using cosine similarity.
    
    :param query: User input query (string).
    :param top_k: Number of top results to return.
    :return: DataFrame with top-k similar products.
    """
    DATA_PATH = "datasets/products_embedding_openai.csv"

    # Encode the query
    query_vec = np.array(get_embedding(query))  # Get the first embedding (query is a single string)
    query_vec = query_vec.flatten()  # Ensure it's a 1D array

    # Load product data and embeddings
    product_vectors, df = convert_numpy(DATA_PATH)
    
    # Compute cosine similarity
    similarities = cosine_similarity([query_vec], product_vectors)[0]

    # Add similarity scores to the DataFrame and sort by similarity
    df["similarity"] = similarities
    results = df.sort_values(by="similarity", ascending=False).head(top_k)

    return results[["name", "price", "similarity", "information_product", "url"]]


# if __name__ == "__main__":
#     # # Specify the column to embed (e.g., "name")
#     # input_file_path = "datasets/products_embedding_openai.csv"  
#     # output_file_path = "datasets/products_embedding_openai.csv"  
#     # column_to_embed = "information_product"
#     # # Process the CSV file
#     # embed_data(input_file_path, output_file_path, column_to_embed)
    
#     # # Test embedding 
#     # query_vec = get_embedding("hello ")
#     # print("😂 ", np.array(query_vec).shape)


#     # User input query
#     user_input = "Tôi muốn mua BQ SP VA85-386"

#     # Search for similar products
#     top_results = search_similar_products(user_input, top_k=10)

#     # Print results
#     # print(f"\n🔎 Kết quả cho truy vấn: \"{user_input}\"\n")
#     for i, row in top_results.iterrows():
#         print(f"👟 {row['name']} - Giá: {int(row['price']):,} VND - Similarity: {row['similarity']:.4f}")
#         print(f"📝 {row['information_product'][:100]}...")
#         print(f"🔗 {row['url']}\n")

#     # Save results to a temporary CSV file
#     top_results.to_csv("temp.csv", index=False)