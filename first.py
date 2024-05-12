from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain.schema import Document

#wird umgeschrieben

# A list of Documents
documents = [
    Document(
        page_content="Text to be indexed",
        metadata={"source": "local"}
    )
]

import os
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Service used to create the embeddings
embedding_provider = OpenAIEmbeddings(
    openai_api_key=openai_api_key
)

new_vector = Neo4jVector.from_documents(
    documents,
    embedding_provider,
    url="bolt://34.234.223.41:7687",
    username="neo4j",
    password="east-rubber-retailer",
    index_name="myVectorIndex",
    node_label="Chunk",
    text_node_property="text",
    embedding_node_property="embedding",
    create_id_index=True,
)