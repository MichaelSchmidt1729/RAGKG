#wird umgeschrieben. siehe genai-stack chains.py loader.py bot.py utils.py
#ToDo aufteilen auf mehrere Dateien oder jupyterNB
# https://neo4j.com/developer-blog/tagged/genai/
# https://blog.gopenai.com/improved-rag-with-llama3-and-ollama-c17dc01f66f6
# https://neo4j.com/developer-blog/construct-knowledge-graphs-unstructured-text/

import os
import re
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.document_loaders import WikipediaLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
url = os.getenv('NEO4J_URI')
username = os.getenv('NEO4J_USERNAME')
password = os.getenv('NEO4J_PASSWORD')

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def bert_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

raw_documents = WikipediaLoader(query="Leonhard Euler").load()
text_splitter = RecursiveCharacterTextSplitter(
          chunk_size = 200,
          chunk_overlap  = 20,
          length_function = bert_len,
          separators=['\n\n', '\n', ' ', ''],
      )

# Instantiate Neo4j vector from documents
neo4j_vector = Neo4jVector.from_documents(
    documents,
    OpenAIEmbeddings(),
    url=url,
    username=username,
    password=password
)

query = "Who were the siblings of Leonhard Euler?"
vector_results = neo4j_vector.similarity_search(query, k=2)
for i, res in enumerate(vector_results):
    print(res.page_content)
    if i != len(vector_results)-1:
        print()
vector_result = vector_results[0].page_content


from langchain.chat_models import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph

graph = Neo4jGraph(
    url=url, username=username, password=password
)

print(graph.schema)

chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(temperature=0), graph=graph, verbose=True
)

graph_result = chain.run("Who were the siblings of Leonhard Euler?")


import json
import sagemaker
import boto3
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri

try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client('iam')
    role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']

hub = {
    'HF_MODEL_ID':'mistralai/Mistral-7B-Instruct-v0.1',
    'SM_NUM_GPUS': json.dumps(1)
}

huggingface_model = HuggingFaceModel(
    image_uri=get_huggingface_llm_image_uri("huggingface",version="1.1.0"),
    env=hub,
    role=role,
)


mistral7b_predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.4xlarge",
    container_startup_health_check_timeout=300,
)

query = "Who were the siblings of Leonhard Euler?"
final_prompt = f"""You are a helpful question-answering agent. Your task is to analyze
and synthesize information from two sources: the top result from a similarity search
(unstructured information) and relevant data from a graph database (structured information).
Given the user's query: {query}, provide a meaningful and efficient answer based
on the insights derived from the following data:

Unstructured information: {vector_result}.
Structured information: {graph_result}.
"""

print(final_prompt)

response = mistral7b_predictor.predict({
    "inputs": final_prompt,
})

print(re.search(r"Answer: (.+)", response[0]['generated_text']).group(1))