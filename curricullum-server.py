from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
from chromadb import PersistentClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import torch
from torch import Tensor
import torch.nn.functional as F

app = FastAPI()

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize the bge-m3 tokenizer and model for embeddings (for query embedding)
embedding_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
embedding_model = AutoModel.from_pretrained('BAAI/bge-m3').to(device)

# Initialize the bge-reranker-v2-m3 tokenizer and model for reranking
reranker_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
reranker_model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3').to(device)

def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Pool the hidden states by taking their average."""
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def generate_embeddings(texts: list, batch_size: int) -> list:
    """Generate embeddings for the query text."""
    embeddings_list = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_dict = embedding_tokenizer(batch_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

        with torch.no_grad():
            outputs = embedding_model(**batch_dict)
            embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
        embeddings_list.extend(embeddings.detach().cpu().numpy().tolist())

    return embeddings_list

def query_with_reranker(query: str, k: int, batch_size: int, dbname: str = "my_text_collection") -> list:
    """
    Query the Chroma DB for a given text, rerank the top results, and return them sorted by relevance.
    """
    # Generate embedding for the query
    query_embedding_list = generate_embeddings([query], batch_size)

    # Connect to Chroma DB, ensuring it's persisted correctly
    persist_directory = "curricular_database"
    client = PersistentClient(path=persist_directory)

    # Debug: Print persist directory and available collections
    print(f"Persist directory: {persist_directory}")
    collections = client.list_collections()
    print(f"Available collections: {[col.name for col in collections]}")

    # Check if the collection exists
    if dbname not in [col.name for col in collections]:
        raise ValueError(f"Collection '{dbname}' not found in the database. Available collections: {[col.name for col in collections]}")

    # Get the collection
    collection = client.get_collection(dbname)
    
    # Perform the query
    result = collection.query(query_embeddings=query_embedding_list, n_results=k)

    # Debug: Print the structure of the result
    print("Query result structure:", result.keys())

    # Prepare pairs for reranking
    pairs = [[query, metadata['text']] for metadata in result['metadatas'][0]]

    # Rerank using bge-reranker-v2-m3
    with torch.no_grad():
        inputs = reranker_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
        scores = reranker_model(**inputs, return_dict=True).logits.view(-1,).float()

    # Create a list of dictionaries with text and score
    reranked_results = [
        {"text": result['metadatas'][0][i]['text'], "score": scores[i].item()}
        for i in range(len(result['metadatas'][0]))
    ]

    # Sort the results by score in descending order
    reranked_results.sort(key=lambda x: x["score"], reverse=True)

    return reranked_results

class QueryRequest(BaseModel):
    query: str
    top_n: int = 5

class QueryResult(BaseModel):
    text: str
    score: float

class QueryResponse(BaseModel):
    results: List[QueryResult]

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    try:
        results = query_with_reranker(request.query, k=request.top_n, batch_size=1, dbname="my_curricula_hh")
        return QueryResponse(results=[QueryResult(text=res["text"], score=res["score"]) for res in results])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8020)