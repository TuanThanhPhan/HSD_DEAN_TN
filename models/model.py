import torch
import torch.nn as nn
from transformers import AutoModel

class HybridHateSpeechModel(nn.Module):

    def __init__(self, phobert_path, char_vocab_size, num_labels = 3):
        super().__init__()

        # ===== PhoBERT =====
        self.phobert = AutoModel.from_pretrained(phobert_path)
        
        # ===== CharCNN =====
        # Embedding ký tự 64 chiều
        self.char_embedding = nn.Embedding(char_vocab_size, 64)
        # Sử dụng Multi-scale CNN (kernel 2,3,4) là cách tốt nhất để bắt teencode
        self.convs = nn.ModuleList([
            nn.Conv1d(64, 64, kernel_size=k, padding=k//2) 
            for k in [2, 3, 4]
        ])
        # Tổng output CNN là (3 * 64), nén về 128
        self.char_fc = nn.Linear(64 * 3, 128)

        # ===== Fusion token-level =====
        self.fusion = nn.Linear(768 + 128, 768)

        # ==== BiLSTM ====
        self.bilstm = nn.LSTM(
            input_size=768,
            hidden_size=384,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        # ==== Classifier ====
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.Mish(), # Dùng Mish activation tốt hơn ReLU
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.Mish(),
            nn.Linear(64, num_labels)
        )
    
    def forward(self, input_ids, attention_mask, char_input):
        # --- PhoBERT  ---
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        bert_out = outputs.last_hidden_state  # [B, S, 768]

        # ===== CharCNN =====
        B, S, L = char_input.shape

        char_x = self.char_embedding(char_input)
        char_x = char_x.view(B * S, L, 64).transpose(1, 2)

        conv_results = []
        for conv in self.convs:
            c = torch.nn.functional.mish(conv(char_x))
            c, _ = torch.max(c, dim=2)
            conv_results.append(c)

        char_feat = torch.cat(conv_results, dim=1)
        char_feat = torch.nn.functional.mish(self.char_fc(char_feat))
        char_feat = char_feat.view(B, S, -1)  # [B, S, 128]

        # ===== Fusion token-level =====
        combined = torch.cat([bert_out, char_feat], dim=2)  # [B, S, 896]
        fused = torch.nn.functional.mish(self.fusion(combined))  # [B, S, 768]

        # ===== BiLSTM =====
        lstm_out, _ = self.bilstm(fused)  # [B, S, 768]

        # ===== Mean pooling =====
        mask = attention_mask.unsqueeze(-1).float()
        summed = torch.sum(lstm_out * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = summed / counts  # [B, 256]

        return self.classifier(pooled)