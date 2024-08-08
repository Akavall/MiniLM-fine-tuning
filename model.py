
import torch.nn as nn

class EmbeddingModel(nn.Module):
    def __init__(self, base_model, output_dim=384):
        super(EmbeddingModel, self).__init__()
        self.base_model = base_model
        self.projection = nn.Linear(384, output_dim) #384 is the number of params that minilm has 

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs[0][:, 0, :]  
        projected_output = self.projection(cls_output)  
        return projected_output