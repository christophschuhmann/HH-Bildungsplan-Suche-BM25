# Import necessary libraries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import bm25s

# Initialize FastAPI app
app = FastAPI()

# Define the path to the BM25 index
INDEX_NAME = "Lehrpläne_HH"

# Load the BM25 index
retriever = bm25s.BM25.load(INDEX_NAME, load_corpus=True)

# Define request model
class QueryRequest(BaseModel):
    query: str
    top_n: int = 5

# Define response models
class QueryResult(BaseModel):
    text: str
    score: float

class QueryResponse(BaseModel):
    results: List[QueryResult]

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    try:
        # Tokenize the query
        query_tokens = bm25s.tokenize(request.query)
        
        # Perform the search
        results, scores = retriever.retrieve(query_tokens, k=request.top_n)
        
        # Prepare the response
        query_results = []
        for i in range(results.shape[1]):
            doc, score = results[0, i], scores[0, i]
            query_results.append(QueryResult(text=doc['text'], score=float(score)))
        
        return QueryResponse(results=query_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8020)
