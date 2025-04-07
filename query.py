from flask import Flask, request, jsonify
import pandas as pd
import json
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask
app = Flask(__name__)

# Gemini API configuration
genai.configure(api_key="AIzaSyDtY_3u6RvOWUebu54gDtgNmNxQKv1gh0Y")
model = genai.GenerativeModel("gemini-1.5-flash")

# Load your assessments JSON data
with open("shl_assessments_complete.json") as f:
    data = json.load(f)
    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, dict) and "assessments" in data:
        df = pd.DataFrame(data["assessments"])
    else:
        df = pd.DataFrame()

# Fallback: ensure necessary columns exist
for col in ['remote_testing', 'adaptive_irt', 'test_type', 'description', 'name', 'url', 'duration']:
    if col not in df.columns:
        df[col] = 'N/A'

# Gemini prompt helper
def query_gemini(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return prompt

# Recommendation logic
def get_recommendations(query, top_k=10):
    gemini_prompt = f"Extract important keywords or skills from: '{query}'."
    enhanced_query = query + " " + query_gemini(gemini_prompt)

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df["description"].fillna(""))
    query_vec = tfidf.transform([enhanced_query])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    df["similarity"] = similarity_scores
    top_results = df.sort_values(by="similarity", ascending=False).head(top_k)

    return top_results[["name", "url", "remote_testing", "adaptive_irt", "duration", "test_type"]]

# GET API endpoint
@app.route("/recommend", methods=["GET"])
def recommend_endpoint():
    query = request.args.get("query", "")
    top_k = request.args.get("top_k", default=10, type=int)

    if not query:
        return jsonify({"error": "Missing 'query' parameter"}), 400

    results = get_recommendations(query, top_k)
    return jsonify(results.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
