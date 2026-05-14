from flask import Flask, render_template, request
import requests
import os

app = Flask(__name__)

API_URL = "https://api-inference.huggingface.co/models/fa22-bcs-102/news-classifier"
HF_TOKEN = os.environ.get("HF_TOKEN")

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

labels_map = {
    "LABEL_0": "WORLD",
    "LABEL_1": "SPORTS",
    "LABEL_2": "BUSINESS",
    "LABEL_3": "SCI/TECH"
}

def predict_news(text):
    payload = {"inputs": text}
    response = requests.post(API_URL, headers=headers, json=payload)
    result = response.json()
    label = result[0][0]['label']
    return labels_map[label]

@app.route("/", methods=["GET","POST"])
def home():
    prediction = ""
    if request.method == "POST":
        news = request.form["news"]
        prediction = predict_news(news)
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run()