import requests

BASE_URL = "http://127.0.0.1:8000"

# 1️⃣ Test root endpoint
resp = requests.get(f"{BASE_URL}/")
print("Root endpoint:", resp.json())

# 2️⃣ Test /query endpoint
query_payload = {
    "query": "What is the main topic of the PDF?",
    "top_k": 3
}

resp = requests.post(f"{BASE_URL}/query", json=query_payload)
print("/query endpoint:", resp.json())

# 3️⃣ Test /flashcards endpoint
resp = requests.get(f"{BASE_URL}/flashcards")
print("/flashcards endpoint:", resp.json())
