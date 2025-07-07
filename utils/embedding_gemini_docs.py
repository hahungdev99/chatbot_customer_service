from importlib.resources import contents
import os
from dotenv import load_dotenv
from google import genai
import pdfplumber
import pandas as pd
import ast
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables from .env file
load_dotenv()

# Replace these with your actual values
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.
    
    :param pdf_path: Path to the PDF file.
    :return: A list of extracted text chunks.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text_chunks = []
            for page in pdf.pages:
                text = page.extract_text()
                text_chunks.append(text)
            return text_chunks
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return []

def get_embedding(text):
    """
    Generate an embedding for the given text using the Gemini embedding model.
    
    :param text: The input text to embed.
    :return: A list representing the embedding vector.
    """
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        # Call Gemini to generate the embedding
        response = client.models.embed_content(
            model="text-embedding-004",  # Replace with the actual Gemini embedding model name
            contents=text
        )

        # Extract the embedding vector
        embedding = response.embeddings[0].values
        return embedding
    except Exception as e:
        print(f"😢 Error generating embedding for '{text}': {str(e)}")
        return None

def chunk_text(text, max_chunk_size=500, overlap=100):
    """
    Split text into overlapping chunks of a specified maximum size.
    
    :param text: Input text to chunk.
    :param max_chunk_size: Maximum size (in characters) of each chunk.
    :param overlap: Number of overlapping characters between consecutive chunks.
    :return: List of overlapping text chunks.
    """
    chunks = []
    start = 0
    # print('😢 len text ', len(text))
    while start < len(text):
        # Define the end index of the current chunk
        end = min(start + max_chunk_size, len(text))
        
        # Append the current chunk
        chunks.append(text[start:end].strip())
        
        # Move the start index forward, but overlap with the previous chunk
        start = end - overlap if end < len(text) else end

    return chunks

def process_pdf_and_save_embeddings(pdf_path, output_csv_path, max_chunk_size=5000, overlap=100):
    """
    Process a PDF file, generate embeddings, and save the results to a CSV file.
    
    :param pdf_path: Path to the PDF file.
    :param output_csv_path: Path to save the output CSV file with embeddings.
    :param max_chunk_size: Maximum size (in characters) of each chunk.
    :param overlap: Number of overlapping characters between consecutive chunks.
    """
    # Extract text from the PDF
    text_chunks = extract_text_from_pdf(pdf_path)

    # Initialize a list to store results
    results = []

    # Generate embeddings for each text chunk
    for i, text in enumerate(text_chunks):
        # Chunk the text with overlap
        chunks = chunk_text(text, max_chunk_size=max_chunk_size, overlap=overlap)

        print('😢 chunks ', chunks)
        for chunk in chunks:
            embedding = get_embedding(chunk)
            if embedding is not None:
                results.append({
                    "page_number": i + 1,
                    "text": chunk,
                    "stores_embedding": [embedding] 
                })

    # Save the results to a CSV file
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"Embeddings saved to: {output_csv_path}")


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

    df["stores_embedding"] = df["stores_embedding"].apply(parse_embedding)
    df = df[df["stores_embedding"].notnull()]  # Drop rows with invalid embeddings

    # Convert to numpy array and validate shape
    product_vectors = np.array(df["stores_embedding"].tolist())
    assert len(product_vectors.shape) == 2, "product_vectors must have 2 dimensions (N, D)"

    return product_vectors, df

# Search for similar products
def search_docs_gemini(query: str, top_k=3):
    """
    Search for products similar to the query using cosine similarity.
    
    :param query: User input query (string).
    :param top_k: Number of top results to return.
    :return: DataFrame with top-k similar products.
    """
    pd.set_option('display.max_colwidth', None)
    DATA_PATH = "datasets/stores_embedding_gemini.csv"

    # Encode the query
    query_vec = np.array(get_embedding(query))  # Get the first embedding (query is a single string)
    query_vec = query_vec.flatten()  # Ensure it's a 1D array

    # print("😂 query vec ", len(query_vec))

    # Load product data and embeddings
    product_vectors, df = convert_numpy(DATA_PATH)
    
    # Compute cosine similarity
    similarities = cosine_similarity([query_vec], product_vectors)[0]

    # Add similarity scores to the DataFrame and sort by similarity
    df["similarity"] = similarities
    results = df.sort_values(by="similarity", ascending=False).head(top_k)

    return results[["page_number", "text"]]



# if __name__ == "__main__":
#     # # Specify the PDF file path and output CSV path
#     # pdf_path = "datasets/data_shopbq.pdf"
#     # output_csv_path = "datasets/stores_embedding_gemini_update.csv"

#     # # Process the PDF and save embeddings
#     # process_pdf_and_save_embeddings(pdf_path, output_csv_path)


#     # user_input = "gần đây có khuyến mãi gì ko vậy ta"
#     # user_input = "Điều kiện đổi trả như thế nào vậy"
#     user_input = "bao lâu thì giao hàng vậy"
#     # user_input = "thời gian giao hàng"
#     # user_input = "shop có cửa hàng ở Quãng Ngãi ko"
#     # Search for similar products
#     # top_results = search_similar_products_gemini(new_user_input, top_k=5)
#     top_results = search_docs_gemini(user_input, top_k=3)
#     # print(top_results)
#     # Print results
#     # print(f"\n🔎 Kết quả cho truy vấn: \"{user_input}\"\n")
#     for i, row in top_results.iterrows():
#         print(f"👌 {row['text']}")

