import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm


from config import MODELS_DIR, ASSETS_DIR
FULL_DATASET = ASSETS_DIR / "hackaton_advertisements_with_id.csv"
RESULT_CSV   = ASSETS_DIR / "full_dataset_with_roberta_features.csv"
FITTED_MODEL = MODELS_DIR / "roberta_student_weights.pth"
MODEL_NAME = "xlm-roberta-base"
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiTaskRoBERTa(nn.Module):
    def __init__(self):
        super(MultiTaskRoBERTa, self).__init__()
        self.roberta = AutoModel.from_pretrained(MODEL_NAME)
        hidden_size = self.roberta.config.hidden_size
        self.head_cosmetic = nn.Linear(hidden_size, 3)
        self.head_functional = nn.Linear(hidden_size, 2)
        self.head_completeness = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids,
                               attention_mask=attention_mask
                            )
        pooled_output = outputs.pooler_output
        return (
            self.head_cosmetic(pooled_output),
            self.head_functional(pooled_output),
            self.head_completeness(pooled_output)
        )


class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }


def run_fast_inference():
    print("Full dataset loading")
    df = pd.read_csv(FULL_DATASET)

    texts = df["description"].fillna("").astype(str).tolist()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = InferenceDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Loading fitted model...")
    model = MultiTaskRoBERTa()
    model.load_state_dict(torch.load(FITTED_MODEL, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    all_results = []

    print(f"Starting labelling {len(texts)} ..")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)

            logits_cosm, logits_func, logits_comp = model(
                input_ids, attention_mask)

            probs_cosm = F.softmax(logits_cosm, dim=1).cpu().numpy()
            probs_func = F.softmax(logits_func, dim=1).cpu().numpy()
            probs_comp = F.softmax(logits_comp, dim=1).cpu().numpy()

            for i in range(len(input_ids)):
                all_results.append({
                    "cosm_prob_0": probs_cosm[i][0],
                    "cosm_prob_1": probs_cosm[i][1],
                    "cosm_prob_2": probs_cosm[i][2],
                    "func_prob_0": probs_func[i][0],
                    "func_prob_1": probs_func[i][1],
                    "comp_prob_0": probs_comp[i][0],
                    "comp_prob_1": probs_comp[i][1]
                })

    print("\nSaving results...")
    results_df = pd.DataFrame(all_results)

    final_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)

    final_df.to_csv(RESULT_CSV, index=False)
    print("Features saved 'full_dataset_with_roberta_features.csv'")


if __name__ == "__main__":
    run_fast_inference()
