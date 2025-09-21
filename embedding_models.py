"""Embedding Models"""

import os
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from langchain_voyageai import VoyageAIEmbeddings

load_dotenv()

documents = [
    'Virat kohli is the best player with 100 runs',
    'Rohit sharma is the best player with 50 runs',
    'Rishabh Pant is the best player with 75 runs',
    "Dhoni is the best player with 150 runs"
]

QUERY = "Who is the best player with 150 runs?"


embeddings = VoyageAIEmbeddings(
    voyage_api_key=os.getenv("VOYAGE_API_KEY"), model="voyage-law-2"
)

document_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(QUERY)

similarity_scores = cosine_similarity(
    [query_embedding], document_embeddings)[0]

index, score = sorted(list(enumerate(similarity_scores)),
                      key=lambda x: x[1], reverse=True)[0]

print(f"Most similar document: {documents[index]} with score: {score}")
