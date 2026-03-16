import pickle
import torch
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

import config

from utils.dataloader import ViHSDDataset
from utils.cleantext import clean_text_pipeline

from models.hybrid_model import HybridHateSpeechModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

dev_df = pd.read_csv(config.DEV_PATH)

dev_df["free_text"] = dev_df["free_text"].apply(clean_text_pipeline)

with open(f"{config.SAVE_DIR}/{config.CHAR_VOCAB_FILE}", "rb") as f:
    char_to_idx = pickle.load(f)

dataset = ViHSDDataset(
    dev_df.free_text.values,
    dev_df.label_id.values,
    tokenizer,
    config.MAX_LEN,
    char_to_idx
)

loader = DataLoader(dataset, batch_size=config.BATCH_SIZE)

model = HybridHateSpeechModel(
    config.MODEL_NAME,
    len(char_to_idx)+2
)

model.load_state_dict(
    torch.load(f"{config.SAVE_DIR}/{config.MODEL_FILE}", map_location=device)
)

model.to(device)
model.eval()

preds = []

with torch.no_grad():

    for batch in loader:

        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        char_in = batch["char_input"].to(device)

        logits = model(input_ids, mask, char_in)

        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())

print(classification_report(dev_df.label_id.values, preds))