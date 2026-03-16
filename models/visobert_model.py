import torch
import torch.nn as nn
from transformers import AutoModel

class ViSoBERTModel(nn.Module):
    def __init__(self, model_path="vinai/visobert-base", num_labels=3):
        super(ViSoBERTModel, self).__init__()

        self.visobert = AutoModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask, char_input=None):
        outputs = self.visobert(input_ids=input_ids, attention_mask=attention_mask)

        # Dùng Mean pooling thay CLS
        # hidden states của tất cả token
        last_hidden_state = outputs.last_hidden_state

        # mở rộng attention_mask để nhân với hidden_state
        mask = attention_mask.unsqueeze(-1).float()

        # mean pooling (bỏ padding)
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled_output = summed / counts

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits