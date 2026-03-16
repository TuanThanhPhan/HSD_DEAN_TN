import torch
import torch.nn as nn
from transformers import AutoModel


class HybridHateSpeechModel(nn.Module):

    def __init__(self, phobert_path, char_vocab_size, hidden_dim=128):
        super().__init__()

        # ===== PhoBERT =====
        self.phobert = AutoModel.from_pretrained(phobert_path)
        self.dropout = nn.Dropout(0.1)

        # ===== CharCNN =====
        # Embedding ký tự 30 chiều
        self.char_embedding = nn.Embedding(char_vocab_size, 30)
        # 30 kernel kích thước 3
        self.char_cnn = nn.Conv1d(
            in_channels=30,
            out_channels=30,
            kernel_size=3,
            padding=1
        )
        # Mean pooling
        self.char_pool = nn.AdaptiveAvgPool1d(1)
        # Fully connected → 128
        self.char_fc = nn.Linear(30, 128)

        # ===== Combine =====
        # PhoBERT (2304) + CharCNN (128)
        input_lstm_dim = 2304 + 128

        self.bilstm = nn.LSTM(
            input_lstm_dim,
            hidden_dim,
            bidirectional=True,
            batch_first=True
        )

        # ===== Classifier =====
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)
        )

    def forward(self, input_ids, attention_mask, char_input):

        # ===== PhoBERT =====
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        hidden_states = outputs.hidden_states

        # nối 3 lớp cuối
        bert_out = torch.cat(
            (hidden_states[-1], hidden_states[-2], hidden_states[-3]),
            dim=-1
        )

        bert_out = self.dropout(bert_out)
        # shape: [batch, seq_len, 2304]

        # ===== CharCNN =====
        # char_input: [B, S, L]
        B, S, L = char_input.shape

        char_x = self.char_embedding(char_input)
        # [B, S, L, 30]

        char_x = char_x.view(B * S, L, 30)
        # [B*S, L, 30]

        char_x = char_x.transpose(1, 2)
        # [B*S, 30, L]

        char_x = torch.relu(self.char_cnn(char_x))
        # [B*S, 30, L]

        char_x = self.char_pool(char_x)
        # [B*S, 30, 1]

        char_x = char_x.squeeze(-1)
        # [B*S, 30]

        char_x = self.char_fc(char_x)
        # [B*S, 128]

        char_x = char_x.view(B, S, 128)
        # [B, S, 128]

        # ===== Combine =====
        combined = torch.cat((bert_out, char_x), dim=2)
        # [B, S, 2432]

        # ===== BiLSTM =====
        lstm_out, _ = self.bilstm(combined)

        # ===== Mean Pooling =====
        final_feature = torch.mean(lstm_out, dim=1)

        logits = self.classifier(final_feature)

        return logits