from qdrant_client import QdrantClient, models

# I'm not atually going ot run this code, I just want a record of these idioms
client = QdrantClient("localhost:6333")
client.create_collection(
    collection_name="{collection_name}",
    vectors_config = models.VectorParams(size=767, distance=models.Distance.COSINE),
    optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
)

# Upload many vectors

client.update_collection(
    collection_name="{collection_name}",
    optimizers_config=models.OptimizersConfigDiff(indexing_threshold=20000),
)

# Consider uploading directly to disk using memmap support

# Parallel upload into multiple shards
client.create_collection(
    collection_name="{collection_name}",
    vectors_config=models.VectorParams(size=767, distance=models.Distance.COSINE),
    shards_count=2,
)
