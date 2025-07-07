from pydantic.v1 import BaseModel, Field, validator
from typing import Any, List
from pymongo import MongoClient
import logging
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import ssl 

# Load cert to work with huggingface incase have issue network
os.environ["REQUESTS_CA_BUNDLE"] = "ca-certificates.crt"
ssl._create_default_https_context = ssl._create_unverified_context

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv()

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

class MongoDBVectorSearch:
    def __init__(self, mongo_uri, database_name, collection_name, model_name="bkai-foundation-models/vietnamese-bi-encoder"):
        """
        Initialize the MongoDBVectorSearch class.

        Args:
        mongo_uri (str): MongoDB connection string.
        database_name (str): Name of the database.
        collection_name (str): Name of the collection.
        model_name (str): Name of the embedding model.
        """
        self.client = MongoClient(mongo_uri)
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]
        self.config = EmbeddingConfig(name=model_name)
        self.model = SentenceTransformerEmbedding(self.config)

    def get_embedding(self, text):
        """
        Generate an embedding for the given text.

        Args:
        text (str): Input text.

        Returns:
        list: Embedding as a list of floats.
        """
        embedding = self.model.encode(text)
        return embedding.flatten().tolist()

    def vector_search(self, user_query, limit=4):
        """
        Perform a vector search in the MongoDB collection based on the user query.

        Args:
        user_query (str): The user's query string.
        limit (int): Number of results to return.

        Returns:
        list: A list of matching documents.
        """
        # Generate embedding for the user query
        query_embedding = self.get_embedding(user_query)
        print("😂😂😂 query_embedding ", len(query_embedding))
        if not query_embedding:
            return "Invalid query or embedding generation failed."

        # Define the vector search pipeline
        vector_search_stage = {
            "$vectorSearch": {
                "index": "vector_index",  # Replace with your vector index name
                "queryVector": query_embedding,
                "path": "embedding_infor_product",  # Field containing embeddings
                "numCandidates": 100,  # Number of candidates to consider
                "limit": limit,  # Number of results to return
            }
        }

        unset_stage = {
            "$unset": "embedding_infor_product"  # Remove the embedding field from results
        }

        project_stage = {
            "$project": {
                "_id": 0,  # Exclude the MongoDB ID
                "name": 1,  # Include product name
                "price": 1,  # Include product price
                "url": 1,  # Include product URL
                "score": {"$meta": "vectorSearchScore"}  # Include similarity score
            }
        }

        pipeline = [vector_search_stage, unset_stage, project_stage]

        # Execute the search
        results = self.collection.aggregate(pipeline)
        return list(results)

# === Configuration ===
MONGO_URI = os.getenv("MONGODB_URI") 
DATABASE_NAME = os.getenv("DATABASE_NAME") 
COLLECTION_NAME = os.getenv("COLLECTION_NAME") 

# === Initialize the Search Class ===
search_engine = MongoDBVectorSearch(MONGO_URI, DATABASE_NAME, COLLECTION_NAME)

# === Test the Search Function ===
if __name__ == "__main__":
    user_input = "Tôi muốn tìm sản phẩm giay sục nữ gót nhon ?"
    results = search_engine.vector_search(user_input, limit=5)

    print(f"\n🔎 Kết quả cho truy vấn: \"{user_input}\"\n")
    for result in results:
        print(f"👟 {result['name']} - Giá: {result['price']} VND - Similarity Score: {result['score']:.4f}")
        print(f"🔗 {result['url']}\n")