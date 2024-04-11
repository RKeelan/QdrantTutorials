import os
import time

from datasets import load_dataset
from qdrant_client import QdrantClient, models

QDRANT_CLUSTER_URL = "http://localhost:6333"

dataset = load_dataset("Qdrant/arxiv-titles-instructorxl-embeddings", split="train", streaming=True)
dataset_iterator = iter(dataset)
train_dataset = [next(dataset_iterator) for _ in range(60000)]
test_dataset = [next(dataset_iterator) for _ in range(1000)]

client = QdrantClient(QDRANT_CLUSTER_URL)

# client.recreate_collection(
#     collection_name="arxiv-titles",
#     vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
# )
# client.upload_points(
#     collection_name="arxiv-titles",
#     points=[
#         models.PointStruct(
#             id=item["id"],
#             vector=item["vector"],
#             payload=item,
#         )
#         for item in train_dataset
#     ]
# )

while True:
    # I think this is resulting in a WinError
    collection_info = client.get_collection("arxiv-titles")
    if collection_info.status == models.CollectionStatus.GREEN:
        # Green means indexing is complete
        break
    print("Waiting for indexing to complete, status is ", collection_info.status)
    time.sleep(1)

print("Indexing complete; collection is ready for search")

def avg_precision_at_k(k: int):
    precisions = []
    for item in test_dataset:
        print(f"Searching for {item['id']}")
        ann_result = client.search(
            collection_name="arxiv-titles",
            query_vector=item["vector"],
            limit=k,
        )

        knn_result = client.search(
            collection_name="arxiv-titles",
            query_vector=item["vector"],
            limit=k,
            search_params=models.SearchParams(
                exact=True, # Exact rather than approximate search
            ),
        )
    
        ann_ids = set(item.id for item in ann_result)
        knn_ids = set(item.id for item in knn_result)
        precision = len(ann_ids.intersection(knn_ids)) / k
        precisions.append(precision)

    return sum(precisions) / len(precisions)

# print(f"avg(precision@5) = {avg_precision_at_k(5)}")

# Tweak the HNSW parameters
client.update_collection(
    collection_name="arxiv-titles",
    hnsw_config=models.HnswConfigDiff(
        m=32, # Increase the number of edges per nodes from tne default 16
        ef_construct=200, # Increase the number of neighbours from the defualt 100
    )
)

while True:
    collection_info = client.get_collection("arxiv-titles")
    if collection_info.status == models.CollectionStatus.GREEN:
        break
    print(F"Waiting for status ({collection_info.status}) to be GREEN")
    time.sleep(1)

print(f"avg(precision@5) = {avg_precision_at_k(5)}")