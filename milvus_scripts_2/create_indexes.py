from pymilvus import MilvusClient

# Prepare index building params


def create_vector_database(db_name,_collection_name,_index_type,_index_name,_metric_type,_params):
    client = MilvusClient(db_name)
    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="vector", # Name of the vector field to be indexed
        index_type=_index_type, # Type of the index to create
        index_name=_index_name, # Name of the index to create
        metric_type=_metric_type, # Metric type used to measure similarity
        params=_params
    )
    if client.has_collection(collection_name=_collection_name):
        client.drop_collection(collection_name=_collection_name)
    client.create_collection(
        collection_name=_collection_name
    )

    client.create_index(
        collection_name=_collection_name,
        index_params=index_params
    )
    


