#pip install flask sentence-transformers transformers faiss-cpu

from flask import Flask, request, jsonify, render_template
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Initialize Flask app
app = Flask(__name__)

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load cross-encoder model for re-ranking
cross_encoder = SentenceTransformer("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Load small open-source language model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
llm = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

# Dummy financial data & FAISS index
documents = ["Company X reported $10M profit in 2023.", "Revenue growth was 15% in 2022."]
embeddings = embedding_model.encode(documents)
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings, dtype=np.float32))

# Input-side guardrail: Validate financial queries
def validate_query(query):
    if len(query) < 5 or not any(keyword in query.lower() for keyword in ["revenue", "profit", "growth", "earnings"]):
        return False, "Invalid or irrelevant query. Please ask about financial data."
    return True, ""

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    user_query = data.get("query", "").strip()
    
    # Apply input validation
    valid, message = validate_query(user_query)
    if not valid:
        return jsonify({"error": message}), 400
    
    # Convert query to embedding
    query_embedding = embedding_model.encode([user_query])
    D, I = index.search(np.array(query_embedding, dtype=np.float32), k=5)
    retrieved_docs = [documents[i] for i in I[0] if i < len(documents)]
    
    # Re-rank using cross-encoder
    query_pairs = [[user_query, doc] for doc in retrieved_docs]
    scores = cross_encoder.encode(query_pairs)
    ranked_docs = [doc for _, doc in sorted(zip(scores.tolist(), retrieved_docs), reverse=True)]

    
    # Generate response using LLM
    input_text = " \n".join(ranked_docs) + "\n Question: " + user_query
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    outputs = llm.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return jsonify({"answer": response, "retrieved_docs": ranked_docs})

if __name__ == "__main__":
    app.run(debug=True)
