import json
import os
from pprint import pprint

from dotenv import load_dotenv
import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Filter, VectorParams
from tqdm.notebook import tqdm
from sentence_transformers import SentenceTransformer

load_dotenv()

QDRANT_CLUSTER_URL = os.environ.get("QDRANT_CLIENT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION_NAME = "startups"

JSON_FILE = "./data/startups_demo.json"
NP_FILE = "./data/vectors.npy"

# Prepare data and upload it to Qdrant

client = QdrantClient(QDRANT_CLUSTER_URL, api_key=QDRANT_API_KEY)
create_client = not client.collection_exists(COLLECTION_NAME)
if create_client:
    if not os.path.exists(NP_FILE):
        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        df = pd.read_json(JSON_FILE, lines=True)
        vectors = model.encode([row.alt + ". " + row.description for _, row in df.iterrows()], show_progress_bar=True)
        np.save(NP_FILE, vectors, allow_pickle=False)
        print(f"Created data/vectors.npy with shape: {vectors.shape}")
    config = VectorParams(size=384, distance=Distance.COSINE)
    client.create_collection(COLLECTION_NAME, vectors_config=config)
    fd = open(JSON_FILE, "r")
    payload = map(json.loads, fd)
    vectors = np.load(NP_FILE)
    client.upload_collection(
        collection_name=COLLECTION_NAME,
        vectors=vectors,
        payload=payload,
        ids=None, # Vector ids will be assigned automatically
        batch_size=256
    )
else:
    print(f"Collection '{COLLECTION_NAME}' already exists.")

# Build the Search API

class NeuralSearcher:
    def __init__(self, collection_name):
        self.collection_name = collection_name
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        self.qdrant_client = QdrantClient(QDRANT_CLUSTER_URL, api_key=QDRANT_API_KEY)
    
    def search(self, text: str, filter: Filter = None):
        vector = self.model.encode(text).tolist()
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=filter,
            limit=5
        )

        payloads = [hit.payload for hit in search_result]
        return payloads

search = NeuralSearcher(COLLECTION_NAME)
city_of_interest = "Berlin"
city_filter = Filter(**{
    "must": [{
        "key": "city",
        "match": {
            "value": city_of_interest
        }
    }]
})
payloads = search.search("AI startup", filter=city_filter)
pprint(payloads)

# I skipped teh FastAPI stuff, because it's not relevant to me