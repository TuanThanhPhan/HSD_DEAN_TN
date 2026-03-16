from tqdm import tqdm
import torch
from sklearn.metrics import f1_score


class Trainer:

    def __init__(self, model, optimizer, criterion, device):

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self, loader):

        self.model.train()

        total_loss = 0

        for batch in tqdm(loader):

            self.optimizer.zero_grad()

            input_ids = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            char_in = batch["char_input"].to(self.device)
            labels = batch["label"].to(self.device)

            logits = self.model(input_ids, mask, char_in)

            loss = self.criterion(logits, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def eval_epoch(self, loader):

        self.model.eval()

        preds = []
        labels_all = []

        with torch.no_grad():

            for batch in loader:

                input_ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                char_in = batch["char_input"].to(self.device)
                labels = batch["label"].to(self.device)

                logits = self.model(input_ids, mask, char_in)

                pred = torch.argmax(logits, dim=1)

                preds.extend(pred.cpu().numpy())
                labels_all.extend(labels.cpu().numpy())

        return f1_score(labels_all, preds, average="macro")