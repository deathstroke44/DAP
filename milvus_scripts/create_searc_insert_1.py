from pymilvus import MilvusClient

# Prepare index building params
client = MilvusClient("milvus_us_accident.db")
index_params = client.prepare_index_params()



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