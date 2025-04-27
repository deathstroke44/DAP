from pymilvus import MilvusClient

# Prepare index building params
client = MilvusClient("milvus_us_accident_6.db")
index_params = client.prepare_index_params()

index_params.add_index(
    field_name="vector", # Name of the vector field to be indexed
    index_type="HNSW", # Type of the index to create
    index_name="hnsw_1", # Name of the index to create
    metric_type="COSINE", # Metric type used to measure similarity
    params={
        "M": 64, # Maximum number of neighbors each node can connect to in the graph
        "efConstruction": 100 # Number of candidate neighbors considered for connection during index construction
    } # Index building params
)
if client.has_collection(collection_name="collection_6"):
    client.drop_collection(collection_name="collection_6")
client.create_collection(
    collection_name="collection_1",
    index_params=index_params,
    dimension=315
)


client = MilvusClient("milvus_us_accident_7.db")
index_params = client.prepare_index_params()

index_params.add_index(
    field_name="vector", # Name of the vector field to be indexed
    index_type="HNSW", # Type of the index to create
    index_name="hnsw_1", # Name of the index to create
    metric_type="IP", # Metric type used to measure similarity
    params={
        "M": 64, # Maximum number of neighbors each node can connect to in the graph
        "efConstruction": 100 # Number of candidate neighbors considered for connection during index construction
    } # Index building params
)
if client.has_collection(collection_name="collection_7"):
    client.drop_collection(collection_name="collection_7")
client.create_collection(
    collection_name="collection_7",
    index_params=index_params,
    dimension=315
)

