#!/usr/bin/env python
# coding: utf-8

# In[3]:


from transformers import BertTokenizer, BertForNextSentencePrediction
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

def predict_next_sentence(sentence1, sentence2):
    # Tokenize input sentences
    input_ids = tokenizer.encode(sentence1, sentence2, return_tensors='pt')

    # Make prediction
    with torch.no_grad():
        logits = model(input_ids)[0]

    # Compute probability of the next sentence
    probability = torch.softmax(logits, dim=1)[:, 0].item()

    return probability

# Example usage
sentence1 = "Artificial Intelligence has revolutionized healthcare diagnostics."
sentence2 = "Patients can now receive faster and more accurate diagnoses."

probability = predict_next_sentence(sentence1, sentence2)
print(f"Probability of the next sentence: {probability:.4f}")


# In[5]:


model.save_pretrained(r'C:\Users\shash\OneDrive\Documents\LinkedIn_Post\BERT_for_NSP')
tokenizer.save_pretrained(r'C:\Users\shash\OneDrive\Documents\LinkedIn_Post\BERT_for_NSP')


# In[ ]:




