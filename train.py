import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup

import config
from seed import set_seed

from utils.cleantext import clean_text_pipeline
from utils.dataloader import ViHSDDataset
from utils.char_vocab import build_char_vocab

from models.model import HybridHateSpeechModel
from trainer import Trainer


def main():

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(config.SAVE_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    train_df = pd.read_csv(config.TRAIN_PATH)
    dev_df = pd.read_csv(config.DEV_PATH)

    train_df["free_text"] = train_df["free_text"].apply(clean_text_pipeline)
    dev_df["free_text"] = dev_df["free_text"].apply(clean_text_pipeline)

    # ===== build char vocab =====
    char_to_idx = build_char_vocab(train_df.free_text.values)

    with open(os.path.join(config.SAVE_DIR, config.CHAR_VOCAB_FILE), "wb") as f:
        pickle.dump(char_to_idx, f)

    train_dataset = ViHSDDataset(
        train_df.free_text.values,
        train_df.label_id.values,
        tokenizer,
        config.MAX_LEN,
        char_to_idx
    )

    dev_dataset = ViHSDDataset(
        dev_df.free_text.values,
        dev_df.label_id.values,
        tokenizer,
        config.MAX_LEN,
        char_to_idx
    )

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config.BATCH_SIZE)

    model = HybridHateSpeechModel(
        config.MODEL_NAME,
        len(char_to_idx) + 2
    )

    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config.LR)

    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(model, optimizer, criterion, device)

    best_f1 = 0
    patience = 0

    for epoch in range(config.EPOCHS):

        train_loss = trainer.train_epoch(train_loader)
        dev_f1 = trainer.eval_epoch(dev_loader)

        print(f"Epoch {epoch+1}")
        print("Train loss:", train_loss)
        print("Dev F1:", dev_f1)

        if dev_f1 > best_f1:

            best_f1 = dev_f1
            patience = 0

            torch.save(
                model.state_dict(),
                os.path.join(config.SAVE_DIR, config.MODEL_FILE)
            )

        else:
            patience += 1

        if patience >= config.PATIENCE:
            print("Early stopping")
            break


if __name__ == "__main__":
    main()