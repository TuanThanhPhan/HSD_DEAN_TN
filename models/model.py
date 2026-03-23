import torch
import torch.nn as nn
from transformers import AutoModel

# THÊM: Lớp Self-Attention để mô hình tự nhận diện từ khóa quan trọng
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, lstm_outputs, attention_mask):
        # lstm_outputs: [batch, seq_len, hidden_dim]
        # attention_mask: [batch, seq_len]
        energy = self.projection(lstm_outputs).squeeze(-1) # [batch, seq_len]
        energy = energy.masked_fill(attention_mask == 0, -1e9)
        weights = torch.softmax(energy, dim=1) # [batch, seq_len]
        
        # Tính trọng số trung bình có trọng điểm
        outputs = (lstm_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs
    
class HybridHateSpeechModel(nn.Module):

    def __init__(self, phobert_path, char_vocab_size, hidden_dim=128):
        super().__init__()

        # ===== PhoBERT =====
        self.phobert = AutoModel.from_pretrained(phobert_path)
        self.dropout = nn.Dropout(0.2)
        # Thêm lớp Linear hạ chiều PhoBERT từ 2304 xuống 256 (Giúp tín hiệu từ CharCNN không bị lấn át)
        self.reduce_phobert = nn.Linear(2304, 256)
        # ===== CharCNN =====
        # Embedding ký tự 64 chiều
        self.char_embedding = nn.Embedding(char_vocab_size, 64)
        # Sử dụng Multi-scale CNN (kernel 2,3,4,5) là cách tốt nhất để bắt teencode
        self.convs = nn.ModuleList([
            nn.Conv1d(64, 64, kernel_size=k, padding=k//2) 
            for k in [2, 3, 4, 5]
        ])
        # Tổng output CNN là (4 * 64), nén về 128
        self.char_fc = nn.Linear(64 * 4, 128)

        # ==== BiLSTM ====
        # Input: PhoBERT_reduced (256) + Char_features (128) = 384
        self.bilstm = nn.LSTM(
            input_size=384,
            hidden_size=hidden_dim,
            num_layers=2, # Tăng lên 2 lớp LSTM để học sâu hơn
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )

        # THÊM: Lớp Attention thay cho Mean Pooling
        self.attention = SelfAttention(hidden_dim * 2)

        # ==== Classifier ====
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.Mish(), # Dùng Mish activation tốt hơn ReLU
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.Mish(),
            nn.Linear(64, 3)
        )
    
    def forward(self, input_ids, attention_mask, char_input):
        # --- PhoBERT Branch ---
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        all_layers = outputs.hidden_states
        # Ghép 3 lớp cuối để lấy thông tin ngữ nghĩa mạnh nhất
        bert_out = torch.cat((all_layers[-1], all_layers[-2], all_layers[-3]), dim=-1)
        bert_out = self.dropout(bert_out)
        bert_out = torch.nn.functional.mish(self.reduce_phobert(bert_out))

        # --- CharCNN Branch ---
        B, S, L = char_input.shape
        char_x = self.char_embedding(char_input) 
        char_x = char_x.view(B * S, L, -1).transpose(1, 2)

        conv_results = []
        for conv in self.convs:
            c = torch.relu(conv(char_x))
            c, _ = torch.max(c, dim=2) # Max-over-time pooling cho ký tự
            conv_results.append(c)
        
        char_feat = torch.cat(conv_results, dim=1) # [B*S, 150]
        char_feat = torch.relu(self.char_fc(char_feat)) # [B*S, 128]
        char_feat = char_feat.view(B, S, 128) # [B, S, 128]

        # --- Hybrid Combine ---
        combined = torch.cat((bert_out, char_feat), dim=2) # [B, S, 384]
        lstm_out, _ = self.bilstm(combined) # [B, S, 256]

        # Dùng Self-Attention để lấy các đặc trưng quan trọng nhất
        attn_out = self.attention(lstm_out, attention_mask)

        # --- Output ---
        return self.classifier(mean_pooled)