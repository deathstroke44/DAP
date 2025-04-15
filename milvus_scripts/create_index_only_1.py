from pymilvus import MilvusClient

# Prepare index building params
client = MilvusClient("milvus_us_accident.db")
index_params = client.prepare_index_params()

index_params.add_index(
    field_name="vector", # Name of the vector field to be indexed
    index_type="HNSW", # Type of the index to create
    index_name="hnsw_1", # Name of the index to create
    metric_type="L2", # Metric type used to measure similarity
    params={
        "M": 64, # Maximum number of neighbors each node can connect to in the graph
        "efConstruction": 100 # Number of candidate neighbors considered for connection during index construction
    } # Index building params
)
if client.has_collection(collection_name="collection_1"):
    client.drop_collection(collection_name="collection_1")
client.create_collection(
    collection_name="collection_1",
    index_params=index_params,
    dimension=5
)


index_params = client.prepare_index_params()

index_params.add_index(
    field_name="vector", # Name of the vector field to be indexed
    index_type="HNSW", # Type of the index to create
    index_name="hnsw_2", # Name of the index to create
    metric_type="L2", # Metric type used to measure similarity
    params={
        "M": 32, # Maximum number of neighbors each node can connect to in the graph
        "efConstruction": 100 # Number of candidate neighbors considered for connection during index construction
    } # Index building params
)

if client.has_collection(collection_name="collection_2"):
    client.drop_collection(collection_name="collection_2")
client.create_collection(
    collection_name="collection_2",
    index_params=index_params,
    dimension=5
)


index_params = client.prepare_index_params()

index_params.add_index(
    field_name="vector", # Name of the vector field to be indexed
    index_type="DISKANN", # Type of the index to create
    index_name="diskann_1", # Name of the index to create
    metric_type="L2", # Metric type used to measure similarity
    params={
        "search_list": 16
    } # Index building params
)


if client.has_collection(collection_name="collection_3"):
    client.drop_collection(collection_name="collection_3")
client.create_collection(
    collection_name="collection_3",
    index_params=index_params,
    dimension=5
)


index_params = client.prepare_index_params()

index_params.add_index(
    field_name="vector", # Name of the vector field to be indexed
    index_type="DISKANN", # Type of the index to create
    index_name="diskann_2", # Name of the index to create
    metric_type="L2", # Metric type used to measure similarity
    params={
        "search_list": 32
    } # Index building params
)


if client.has_collection(collection_name="collection_4"):
    client.drop_collection(collection_name="collection_4")
client.create_collection(
    collection_name="collection_4",
    index_params=index_params,
    dimension=5
)
