from create_indexes import *

    

create_vector_database("milvus_us_accident-1.db","HNSW","HNSW-1","L2",{"M": 64, "efConstruction": 100})
create_vector_database("milvus_us_accident-2.db","HNSW","HNSW-2","L2",{"M": 32, "efConstruction": 100})
create_vector_database("milvus_us_accident-3.db","IVF_FLAT","IVF_FLAT-1","L2",{"nlist": 1024})
create_vector_database("milvus_us_accident-4.db","IVF_FLAT","IVF_FLAT-1","L2",{"nlist": 512})
create_vector_database("milvus_us_accident-5.db","IVF_FLAT","IVF_PQ-1","L2",{"nlist": 1024,"m":32})
create_vector_database("milvus_us_accident-6.db","HNSW","HNSW-1","COSINE",{"M": 64, "efConstruction": 100})
create_vector_database("milvus_us_accident-7.db","HNSW","HNSW-2","COSINE",{"M": 32, "efConstruction": 100})
create_vector_database("milvus_us_accident-8.db","IVF_FLAT","IVF_FLAT-1","COSINE",{"nlist": 1024})
create_vector_database("milvus_us_accident-9.db","IVF_FLAT","IVF_FLAT-1","COSINE",{"nlist": 512})
create_vector_database("milvus_us_accident-10.db","IVF_FLAT","IVF_PQ-1","COSINE",{"nlist": 1024,"m":32})
create_vector_database("milvus_us_accident-11.db","HNSW","HNSW-1","IP",{"M": 64, "efConstruction": 100})
create_vector_database("milvus_us_accident-12.db","HNSW","HNSW-2","IP",{"M": 32, "efConstruction": 100})
create_vector_database("milvus_us_accident-13.db","IVF_FLAT","IVF_FLAT-1","IP",{"nlist": 1024})
create_vector_database("milvus_us_accident-14.db","IVF_FLAT","IVF_FLAT-1","IP",{"nlist": 512})
create_vector_database("milvus_us_accident-15.db","IVF_FLAT","IVF_PQ-1","IP",{"nlist": 1024,"m":32})

