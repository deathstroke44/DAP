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

search_params = {
    "params": {
        "ef": 10, # Number of neighbors to consider during the search
    }
}
data = [
    {"id": 1, "vector": [0.1, 0.2, 0.3, 0.4, 0.5], "text": 'omi', "subject": "history"},
    {"id": 2, "vector": [0.1, 0.2, 0.3, 0.4, 0.1], "text": 'oni', "subject": "history"},
    {"id": 3, "vector": [0.1, 0.2, 0.3, 0.4, 0.6], "text": 'oxi', "subject": "history"},
]


res = client.insert(collection_name="collection_1", data=data)

print(res)
res = client.search(
    collection_name="collection_1", # Collection name
    data=[[0.1, 0.2, 0.3, 0.4, 0.5]],  # Query vector
    limit=10,  # TopK results to return
    output_fields=["text", "subject"], 
    search_params=search_params
)

print(res)

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

search_params = {
    "params": {
        "ef": 10, # Number of neighbors to consider during the search
    }
}
data = [
    {"id": 1, "vector": [0.1, 0.2, 0.3, 0.4, 0.5], "text": 'omi', "subject": "history"},
    {"id": 2, "vector": [0.1, 0.2, 0.3, 0.4, 0.1], "text": 'oni', "subject": "history"},
    {"id": 3, "vector": [0.1, 0.2, 0.3, 0.4, 0.6], "text": 'oxi', "subject": "history"},
]


res = client.insert(collection_name="collection_2", data=data)

print(res)
res = client.search(
    collection_name="collection_2", # Collection name
    data=[[0.1, 0.2, 0.3, 0.4, 0.5]],  # Query vector
    limit=10,  # TopK results to return
    output_fields=["text", "subject"], 
    search_params=search_params
)

print(res)


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

search_params = {
    "params": {
        "ef": 10, # Number of neighbors to consider during the search
    }
}
data = [
    {"id": 1, "vector": [0.1, 0.2, 0.3, 0.4, 0.5], "text": 'omi', "subject": "history"},
    {"id": 2, "vector": [0.1, 0.2, 0.3, 0.4, 0.1], "text": 'oni', "subject": "history"},
    {"id": 3, "vector": [0.1, 0.2, 0.3, 0.4, 0.6], "text": 'oxi', "subject": "history"},
]


res = client.insert(collection_name="collection_3", data=data)

res = client.search(
    collection_name="collection_3", # Collection name
    data=[[0.1, 0.2, 0.3, 0.4, 0.5]],  # Query vector
    limit=10,  # TopK results to return
    output_fields=["text", "subject"]
)

print(res)


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

search_params = {
    "params": {
        "ef": 10, # Number of neighbors to consider during the search
    }
}
data = [
    {"id": 1, "vector": [0.1, 0.2, 0.3, 0.4, 0.5], "text": 'omi', "subject": "history"},
    {"id": 2, "vector": [0.1, 0.2, 0.3, 0.4, 0.1], "text": 'oni', "subject": "history"},
    {"id": 3, "vector": [0.1, 0.2, 0.3, 0.4, 0.6], "text": 'oxi', "subject": "history"},
]


res = client.insert(collection_name="collection_4", data=data)


res = client.search(
    collection_name="collection_4", # Collection name
    data=[[0.1, 0.2, 0.3, 0.4, 0.5]],  # Query vector
    limit=10,  # TopK results to return
    output_fields=["text", "subject"]
)

print(res)