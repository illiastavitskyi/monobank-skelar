from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

BASE_DIR = r"D:\venvs\venv2\code\test_task_skelar_grandfinal\data\labeled_dataset.csv"
MODEL_NAME = "xlm-roberta-base"
BATCH_SIZE = 16
EPOCHS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Використовуємо пристрій: {DEVICE}")


class MonoBazarDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.texts = df["text"].tolist()
        self.cosmetic = df["cosmetic"].tolist()
        self.functional = df["functional"].tolist()
        self.completeness = df["completeness"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "cosmetic": torch.tensor(self.cosmetic[idx], dtype=torch.long),
            "functional": torch.tensor(self.functional[idx], dtype=torch.long),
            "completeness": torch.tensor(self.completeness[idx],
                                         dtype=torch.long),
        }


class MultiTaskRoBERTa(nn.Module):
    def __init__(self):
        super(MultiTaskRoBERTa, self).__init__()
        self.roberta = AutoModel.from_pretrained(MODEL_NAME)
        hidden_size = self.roberta.config.hidden_size

        self.head_cosmetic = nn.Linear(hidden_size, 3)
        self.head_functional = nn.Linear(hidden_size, 2)
        self.head_completeness = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        return (
            self.head_cosmetic(pooled_output),
            self.head_functional(pooled_output),
            self.head_completeness(pooled_output),
        )


def train():
    print("Loading dataset...")
    df = pd.read_csv(BASE_DIR)

    df = df.dropna(subset=["text"])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = MonoBazarDataset(df, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("Initializating RoBERTa...")
    model = MultiTaskRoBERTa().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    print("Start fitting (Distillation)...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        loop = tqdm(dataloader, leave=True, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for batch in loop:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            logits_cosm, logits_func, logits_comp = model(
                input_ids, attention_mask
            )

            loss_cosm = criterion(logits_cosm, batch["cosmetic"].to(DEVICE))
            loss_func = criterion(logits_func, batch["functional"].to(DEVICE))
            loss_comp = criterion(logits_comp, batch["completeness"].to(DEVICE))

            loss = loss_cosm + loss_func + loss_comp

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

    print("\nFitting completed Saving student model...")
    torch.save(model.state_dict(), "roberta_student_weights.pth")
    print("File roberta_student_weights.pth saved successfully!")


if __name__ == "__main__":
    train()
