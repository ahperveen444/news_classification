import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load trained model & tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("tokenizer")
model = DistilBertForSequenceClassification.from_pretrained("model")

model.eval()

# AG NEWS labels
labels = {
    0: "WORLD 🌍",
    1: "SPORTS ⚽",
    2: "BUSINESS 💰",
    3: "SCI/TECH 💻"
}

# prediction loop
while True:
    text = input("\nEnter news text (or type quit): ")

    if text.lower() == "quit":
        break

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    print("Predicted Category:", labels[prediction])