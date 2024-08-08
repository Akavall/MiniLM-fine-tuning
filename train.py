
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW
from torch import nn
import torch.nn.functional as F

import pandas as pd

from model import EmbeddingModel

import params

class TermSimilarityDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=params.MAX_TOKEN_LENGTH):
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        term_1 = self.data.iloc[idx, 0]
        term_2 = self.data.iloc[idx, 1]
        similarity_score = self.data.iloc[idx, 2]

        inputs = {}
        
        inputs_1 = self.tokenizer(term_1, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        inputs_2 = self.tokenizer(term_2, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")

        inputs["term_1"] = inputs_1
        inputs["term_2"] = inputs_2

        inputs['labels'] = torch.tensor(similarity_score, dtype=torch.float)

        return inputs

tokenizer = AutoTokenizer.from_pretrained('microsoft/MiniLM-L12-H384-uncased')
model = AutoModel.from_pretrained('microsoft/MiniLM-L12-H384-uncased')

dataset = TermSimilarityDataset('train_data.csv', tokenizer)
dataloader = DataLoader(dataset, batch_size=params.TRAINING_BATCH_SIZE, shuffle=True)

eval_dataset = TermSimilarityDataset("eval_data.csv", tokenizer)
eval_dataloader = DataLoader(dataset, batch_size=params.TRAINING_BATCH_SIZE, shuffle=True)

similarity_model = EmbeddingModel(model, output_dim=params.OUTPUT_DIM)

optimizer = AdamW(similarity_model.parameters(), lr=params.LEARNING_RATE)
criterion = nn.MSELoss()

similarity_model.train()
epoch_loss = 0

best_model_score = torch.inf 
for epoch in range(params.N_EPOCHS):  
    eval_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()

        input_ids_1 = batch["term_1"]['input_ids'].squeeze(1)
        attention_mask_1 = batch["term_1"]['attention_mask'].squeeze(1)
        token_type_ids_1 = batch["term_1"]['token_type_ids'].squeeze(1)

        input_ids_2 = batch["term_2"]['input_ids'].squeeze(1)
        attention_mask_2 = batch["term_2"]['attention_mask'].squeeze(1)
        token_type_ids_2 = batch["term_2"]['token_type_ids'].squeeze(1)

        labels = batch['labels']

        embedding_1 = similarity_model(input_ids_1, attention_mask_1, token_type_ids_1)
        embedding_2 = similarity_model(input_ids_2, attention_mask_2, token_type_ids_2)

        estimated_sim = F.cosine_similarity(embedding_1, embedding_2)

        loss = criterion(estimated_sim, labels)
        epoch_loss += loss
        loss.backward()
        optimizer.step()

    print(f'This {epoch + 1}, Loss: {epoch_loss / len(dataset)}')

    similarity_model.eval()
    eval_loss = 0
    with torch.no_grad():
        for batch in eval_dataloader:

            input_ids_1 = batch["term_1"]['input_ids'].squeeze(1)
            attention_mask_1 = batch["term_1"]['attention_mask'].squeeze(1)
            token_type_ids_1 = batch["term_1"]['token_type_ids'].squeeze(1)

            input_ids_2 = batch["term_2"]['input_ids'].squeeze(1)
            attention_mask_2 = batch["term_2"]['attention_mask'].squeeze(1)
            token_type_ids_2 = batch["term_2"]['token_type_ids'].squeeze(1)

            embedding_1 = similarity_model(input_ids_1, attention_mask_1, token_type_ids_1)
            embedding_2 = similarity_model(input_ids_2, attention_mask_2, token_type_ids_2)

            estimated_sim = F.cosine_similarity(embedding_1, embedding_2)

            labels = batch['labels']

            loss = criterion(estimated_sim, labels)
            eval_loss += loss.item()

    avg_eval_loss = eval_loss / len(eval_dataloader)
    print(f'Epoch {epoch + 1}, Evaluation Loss: {avg_eval_loss}')

    if avg_eval_loss < best_model_score:
        print(f"Saving new model...")
        torch.save(similarity_model.state_dict(), 'model.pt')
        avg_eval_loss = best_model_score