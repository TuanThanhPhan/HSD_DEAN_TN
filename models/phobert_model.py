import torch
import torch.nn as nn
from transformers import AutoModel

class PhoBERTModel(nn.Module):
    def __init__(self, model_path="vinai/phobert-base", num_labels=3):
        super(PhoBERTModel, self).__init__()
        self.phobert = AutoModel.from_pretrained(model_path)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.Mish(), # Dùng Mish đồng bộ với Hybrid
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.Mish(),
            nn.Linear(64, num_labels)
        )

    def forward(self, input_ids, attention_mask, char_input=None):
        # char_input để None để đồng bộ hàm gọi với các model khác
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # hidden states của tất cả token
        last_hidden_state = outputs.last_hidden_state

        # Mean pooling (bỏ padding)
        mask = attention_mask.unsqueeze(-1).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled_output = summed / counts

        # Không cần dropout lẻ ở ngoài vì đã có trong Sequential
        return self.classifier(pooled_output)