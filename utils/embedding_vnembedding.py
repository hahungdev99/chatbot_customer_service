from sentence_transformers import SentenceTransformer
import os
import logging
import ssl
import pandas as pd
import ast
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Configure Pandas display options
pd.set_option('display.max_colwidth', None)  # Show full content of columns
pd.set_option('display.max_rows', None)      # Show all rows
pd.set_option('display.width', 1000)         # Set display width for better readability

# === Load model ===
MODEL_NAME = "dangvantuan/vietnamese-embedding"
LOCAL_MODEL_PATH = os.path.join("llm_models", MODEL_NAME)

def load_or_download_model():
    """
    Load the embedding model from local cache or download it if necessary.
    """
    try:
        if os.path.exists(os.path.join(LOCAL_MODEL_PATH)):
            logging.info(f"👍 Loading model from local cache: {LOCAL_MODEL_PATH}")
            return SentenceTransformer(LOCAL_MODEL_PATH)
        else:
            logging.info(f"👍 Downloading model: {MODEL_NAME}")
            model = SentenceTransformer(MODEL_NAME)
            logging.info(f"👍 Model saved to: {LOCAL_MODEL_PATH}")
            return model
    except Exception as e:
        logging.error(f"Failed to load or download model: {e}")
        raise

# Load the embedding model
embedding_model = load_or_download_model()


def embed_data(input_file_path, output_file_path, column_name):
    """
    Embed data from a specified column in a CSV file and save the results to a new file.
    
    :param input_file_path: Path to the input CSV file.
    :param output_file_path: Path to save the output CSV file with embeddings.
    :param column_name: Name of the column to embed.
    """
    # Load the input CSV file
    logging.info(f"📂 Loading data from: {input_file_path}")
    df = pd.read_csv(input_file_path, encoding="utf-8-sig")
    
    # Check if the specified column exists
    if column_name not in df.columns:
        logging.error(f"Column '{column_name}' not found in the input CSV file.")
        return
    
    # Encode the specified column into embeddings
    logging.info(f"Generating embeddings for column: {column_name}")
    df[f"{column_name}_embedding"] = df[column_name].apply(
        lambda x: [embedding_model.encode(x).tolist()] if isinstance(x, str) else None
    )
    
    # Drop rows where embedding generation failed
    df = df[df[f"{column_name}_embedding"].notnull()]
    
    # Save the updated DataFrame to the output CSV file
    logging.info(f"Saving embedded data to: {output_file_path}")
    df.to_csv(output_file_path, index=False, encoding="utf-8-sig")
    logging.info("Embedding process completed successfully.")

# Function to encode text into embeddings
def encode(text):
    """
    Encode text into an embedding vector.
    
    :param text: Input text (string or list of strings).
    :return: List of embedding vectors.
    """
    if isinstance(text, str):
        text = [text]
    return embedding_model.encode(text)

# Convert CSV file's embedding column from string to numpy array
def convert_numpy(DATA_PATH, column_name):
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

    df[f"{column_name}_embedding"] = df[f"{column_name}_embedding"].apply(parse_embedding)
    df = df[df[f"{column_name}_embedding"].notnull()]  # Drop rows with invalid embeddings

    # Convert to numpy array and validate shape
    product_vectors = np.array(df[f"{column_name}_embedding"].tolist())
    logging.info(f"Shape of product_vectors: {product_vectors.shape}")  # Debugging line
    assert len(product_vectors.shape) == 2, "product_vectors must have 2 dimensions (N, D)"

    return product_vectors, df

# Search for similar products
def search_similar_products_vnembedding(column_name, query: str, top_k=5):
    """
    Search for products similar to the query using cosine similarity.
    
    :param query: User input query (string).
    :param top_k: Number of top results to return.
    :return: DataFrame with top-k similar products.
    """
    pd.set_option('display.max_colwidth', None)
    DATA_PATH = "datasets/products_embedding_vnembedding.csv"

    # Encode the query
    query_vec = encode(query)[0]  # Get the first embedding (query is a single string)
    query_vec = query_vec.flatten()  # Ensure it's a 1D array

    # Load product data and embeddings
    product_vectors, df = convert_numpy(DATA_PATH, column_name)
    
    # Compute cosine similarity
    similarities = cosine_similarity([query_vec], product_vectors)[0]

    # Add similarity scores to the DataFrame and sort by similarity
    df["similarity"] = similarities
    results = df.sort_values(by="similarity", ascending=False).head(top_k)

    return results[["name"]]

def search_similar_products_vnembedding_mongodb(column_name, query: str, top_k=5):
        
    MONGO_URI = "mongodb+srv://hunghah:Hung123456@cluster0.ewg2hep.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

    DATABASE_NAME = "chatbot"
    # Thay đổi COLLECTION_NAME nếu bạn dùng tên khác (ví dụ: 'shoe')
    COLLECTION_NAME = "shoe" 
    # Điền đúng tên Vector Search Index bạn đã tạo trên Atlas
    VECTOR_INDEX_NAME = "vector_search" 
    # model_path = os.path.join('llm_models', 'dangvantuan/vietnamese-embedding')
    # model = SentenceTransformer(model_path)
    
    client = None
    try:
        # Kết nối tới MongoDB Atlas
        print("🔄 Đang kết nối tới MongoDB Atlas...")
        client = MongoClient(MONGO_URI)
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        print("✅ Kết nối thành công!")

        # Chuyển câu truy vấn của người dùng thành vector
        query_embedding = embedding_model.encode(query).tolist()

        # Xây dựng pipeline cho Vector Search
        pipeline = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX_NAME,
                    "path": column_name + "_embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 150,
                    "limit": top_k
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "name": 1,
                    "information_product": 1,
                    "score": { "$meta": "vectorSearchScore" }
                }
            }
        ]

        # Thực thi truy vấn
        print(f"\n🔄 Đang tìm kiếm cho truy vấn: '{query}'...")
        results = collection.aggregate(pipeline)
        answer = []
        for doc in results:
            # print(doc)
            answer.append(doc.get("information_product"))
        return answer
    
    except Exception as e:
        print(f"❌ Đã xảy ra lỗi: {e}")

    finally:
        # Đảm bảo client luôn được đóng
        if client:
            client.close()
            print("🔌 Đã đóng kết nối.")


if __name__ == "__main__":
    # # Embedding data
    # input_file_path = "datasets/products_embedding_vnembedding.csv"
    # output_file_path = "datasets/products_embedding_vnembedding.csv"
    # column_name = "information_product"
    # embed_data(input_file_path, output_file_path, column_name)

    # # Test embedding
    # query_vec = encode("hello ")[0]
    # print('😂 ', query_vec.shape)


    # User input query
    user_input = "giày da nam BQ SD 031"
    # column_name = "name"
    column_name = "name"
    
    # # Search for similar products
    # search_csv = search_similar_products_vnembedding(column_name=column_name, query= user_input, top_k=5)
    # for i, row in search_csv.iterrows():
    #     # print(f"👟 {row['name']} - Giá: {int(row['price']):,} VND - Similarity: {row['similarity']:.4f}")
    #     print(f"✔ Tên sản phẩm: {row['name'][:100]}")

    
    print("===================================================")
    search_mongo = search_similar_products_vnembedding_mongodb(column_name, user_input , top_k=5)
    print(search_mongo)
    # for doc in search_mongo:
        # print(f"✔  Tên sản phẩm: {doc.get('name')}")


    # # # Save results to a temporary CSV file
    # # top_results.to_csv("temp.csv", index=False)
