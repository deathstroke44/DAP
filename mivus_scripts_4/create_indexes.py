from pymilvus import MilvusClient,DataType

def create_vector_database(db_name, _index_type, _metric_type, _params):
    client = MilvusClient("scalibility.db")

    # Drop collection if it exists
    if client.has_collection(collection_name='collection_1'):
        client.drop_collection(collection_name='collection_1')

    # Now create a new collection
    
    # Prepare index after the collection is created
    
    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
    )

    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=315)
    schema.add_field(field_name="state", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="label", datatype=DataType.INT32)

    
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",  # Name of the vector field to be indexed
        index_type=_index_type,  # Type of the index to create
        metric_type=_metric_type,  # Metric type used to measure similarity
        params=_params
    )
    client.create_collection(
        collection_name='collection_1',
        schema=schema
    )

    # Now create the index
    client.create_index(
        collection_name='collection_1',
        index_params=index_params
    )
    
create_vector_database("default","IVF_FLAT","L2",{"nlist": 1024})
    
    
