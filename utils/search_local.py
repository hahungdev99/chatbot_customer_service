from pydantic.v1 import BaseModel, Field, validator
from typing import Any, List
from sentence_transformers import SentenceTransformer
import os
import logging
import ssl
import pandas as pd
import ast
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class EmbeddingConfig(BaseModel):
    name: str = Field(..., description="The name of the SentenceTransformer model")

    @validator('name')
    def check_model_name(cls, value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Model name must be a non-empty string")
        return value

from abc import ABC, abstractmethod

class BaseEmbedding(ABC):
    name: str

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def encode(self, text: Any) -> List[float]:
        pass

class SentenceTransformerEmbedding(BaseEmbedding):
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config.name)
        self.config = config
        self.embedding_model = self.load_or_download_model()

    def load_or_download_model(self):
        local_model_path = "bkai-foundation-models/vietnamese-bi-encoder"
        try:
            if os.path.exists(os.path.join(local_model_path, "modules.json")):
                logging.info(f"Loading model from local cache: {local_model_path}")
                return SentenceTransformer(local_model_path)
            else:
                logging.info(f"Downloading model: {self.config.name}")
                model = SentenceTransformer(self.config.name, cache_folder=local_model_path)
                logging.info(f"Model saved to: {local_model_path}")
                return model
        except Exception as e:
            logging.error(f"Failed to load or download model: {e}")
            raise

    def encode(self, text: Any) -> List[float]:
        if isinstance(text, str):
            text = [text]
        return self.embedding_model.encode(text)

### TEXT EMBEDDING QUERY
# === Load model ===
# === Cấu hình ===




# load file .csv and then convert the embedding_name_product to list for query 
def convert_numpy(DATA_PATH):

    # === Load dữ liệu sản phẩm ===
    # print("📂 Loading product data...")
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    # Chuyển cột embedding từ chuỗi sang list số nếu cần
    def parse_embedding(x):
        try:
            emb = ast.literal_eval(x) if isinstance(x, str) else x
            return np.array(emb).flatten()  # Flatten to 1D array
        except:
            return None

    df["embedding_name_product"] = df["embedding_name_product"].apply(parse_embedding)
    df = df[df["embedding_name_product"].notnull()]  # Bỏ dòng lỗi

    # Chuyển thành ma trận numpy và kiểm tra shape
    product_vectors = np.array(df["embedding_name_product"].tolist())
    logging.info(f"Shape of product_vectors: {product_vectors.shape}")  # Debugging line
    assert len(product_vectors.shape) == 2, "product_vectors must have 2 dimensions (N, D)"

    return product_vectors, df


def search_similar_products(query: str, top_k=5):
    
    MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"  
    # DATA_PATH = "datasets/products_with_embeddings_vector_fixed.csv"  
    DATA_PATH = "datasets/products_with_embeddings_vector_only_name.csv"  

    config = EmbeddingConfig(name=MODEL_NAME)
    model = SentenceTransformerEmbedding(config)
    query_vec = model.encode(query)  
    query_vec = query_vec.flatten()  
    
    product_vectors, df = convert_numpy(DATA_PATH)

    similarities = cosine_similarity([query_vec], product_vectors)[0]  
    
    df["similarity"] = similarities
    results = df.sort_values(by="similarity", ascending=False).head(top_k)

    return results[["name", "price", "similarity", "information_product", "url"]]




# if __name__ == "__main__":
#     # user_input = "Tôi muốn tìm sản phẩm giày thể thao nam Slip Ons 2S, bạn có thể giúp tôi tìm kiếm sản phẩm tương tự không ?"
#     user_input = "Giày cỏ nam xỏ chân da êm mềm"
#     top_results = search_similar_products(user_input, top_k=1)

#     print(f"\n🔎 Kết quả cho truy vấn: \"{user_input}\"\n")
#     print((top_results))
#     # top_results.to_csv("temp.csv")
#     for i, row in top_results.iterrows():

#         print(f"👟 {row['name']} - Giá: {int(row['price']):,} VND - Similarity: {row['similarity']:.4f}")
#         print(f"📝 {row['information_product'][:100]}...")
#         print(f"🔗 {row['url']}\n")

