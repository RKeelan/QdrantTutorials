from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
import json

encoder = SentenceTransformer("all-MiniLM-L6-v2") # https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

# Dataset
with open('documents.json', 'r') as f:
    documents = json.load(f)

# Setup Qdrant
qdrant = QdrantClient(":memory:")
qdrant.create_collection(
    collection_name="my_books",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(),
        distance=models.Distance.COSINE,
    ),
)

# Upload data
qdrant.upload_points(
    collection_name="my_books",
    points=[
        models.PointStruct(
            id=idx,
            vector=encoder.encode(doc['description']).tolist(),
            payload=doc
        )
        for idx, doc in enumerate(documents)
    ],
)

# hits = qdrant.search(
#     collection_name="my_books",
#     query_vector=encoder.encode("alien invasion").tolist(),
#     limit=3,
# )

hits = qdrant.search(
    collection_name="my_books",
    query_vector=encoder.encode("alien invasion").tolist(),
    query_filter=models.Filter(
        must=[models.FieldCondition(key="year", range=models.Range(gte=2000))]
    ),
    limit=1,
)


for hit in hits:
    print(hit.payload, "score:", hit.score)