from transformers import BertTokenizer, BertForNextSentencePrediction
from flask import Flask, render_template, request
import torch

app = Flask(__name__)

# Load pre-trained BERT model and tokenizer
model_path = 'C:/Users/shash/OneDrive/Documents/LinkedIn_Post/BERT_for_NSP'
tokenizer_path = 'C:/Users/shash/OneDrive/Documents/LinkedIn_Post/BERT_for_NSP'

tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
model = BertForNextSentencePrediction.from_pretrained(model_path)

def predict_next_sentence(sentence1, sentence2):
    # Tokenize input sentences
    input_ids = tokenizer.encode(sentence1, sentence2, return_tensors='pt')

    # Make prediction
    with torch.no_grad():
        logits = model(input_ids)[0]

    # Compute probability of the next sentence
    probability = torch.softmax(logits, dim=1)[:, 0].item()

    return probability

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text1 = request.form['input_text1']
    input_text2 = request.form['input_text2']
    
    # Process input_text using the loaded BERT model and tokenizer
    probability = predict_next_sentence(input_text1, input_text2)
    return render_template('index.html', prediction=probability)

if __name__ == '__main__':
    app.run(debug=True)
