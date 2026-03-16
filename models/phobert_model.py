import torch
import torch.nn as nn
from transformers import AutoModel

class PhoBERTModel(nn.Module):
    def __init__(self, model_path="vinai/phobert-base", num_labels=3):
        super(PhoBERTModel, self).__init__()
        self.phobert = AutoModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask, char_input=None):
        # char_input để None để đồng bộ hàm gọi với các model khác
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # hidden states của tất cả token
        last_hidden_state = outputs.last_hidden_state

        # tạo mask để bỏ padding
        mask = attention_mask.unsqueeze(-1).float()

        # mean pooling (không tính padding)
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled_output = summed / counts

        pooled_output = self.dropout(pooled_output)

        return self.classifier(pooled_output)