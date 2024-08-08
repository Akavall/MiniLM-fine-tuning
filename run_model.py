
from model import EmbeddingModel
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

import params

def get_single_term_embedding(term, tokenizer, model):
    inputs = tokenizer(term, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    inputs['input_ids'] = inputs['input_ids'].squeeze(1)
    inputs['attention_mask'] = inputs['attention_mask'].squeeze(1)
    if 'token_type_ids' in inputs:
        inputs['token_type_ids'] = inputs['token_type_ids'].squeeze(1)
    
    with torch.no_grad():
        cls_embedding = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], token_type_ids=inputs.get('token_type_ids'))
    return cls_embedding

if __name__ == "__main__":

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('microsoft/MiniLM-L12-H384-uncased')
    base_model = AutoModel.from_pretrained('microsoft/MiniLM-L12-H384-uncased')
    similarity_model = EmbeddingModel(base_model, output_dim=params.OUTPUT_DIM)

    # Load the state dictionary
    similarity_model.load_state_dict(torch.load('model.pt'))
    similarity_model.eval()

    term_1 = "data science"
    term_2 = "skateboarding"

    embedding_1 = get_single_term_embedding(term_1, tokenizer, similarity_model)
    embedding_2 = get_single_term_embedding(term_2, tokenizer, similarity_model)

    estimated_sim = F.cosine_similarity(embedding_1, embedding_2)

    print(f"estimated_sim: {estimated_sim}")
