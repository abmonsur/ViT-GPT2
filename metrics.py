import pandas as pd
import numpy as np
import evaluate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve,
    mean_squared_error, r2_score, matthews_corrcoef
)
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from tqdm import tqdm

df = pd.read_csv("flickr30k_custom_predictions.csv") 
preds = df["predicted"].astype(str).tolist()
refs  = [[r] for r in df["caption"].astype(str).tolist()]

print(f" Loaded {len(preds)} prediction–reference pairs.")

bleu   = evaluate.load("bleu")
meteor = evaluate.load("meteor")

bleu_score   = bleu.compute(predictions=preds, references=refs)
meteor_score = meteor.compute(predictions=preds, references=refs)

print("\n Captioning Evaluation Metrics:")
print(f"BLEU-1..4 : {bleu_score['precisions']}")
print(f"BLEU      : {bleu_score['bleu']:.4f}")
print(f"METEOR    : {meteor_score['meteor']:.4f}")

individual_meteor_scores = []
for p, r in tqdm(zip(preds, refs), total=len(preds), desc="Computing individual Meteor scores"):
    s = meteor.compute(predictions=[p], references=[r])['meteor']
    individual_meteor_scores.append(s)

threshold = 0.3  # caption considered correct if BLEU ≥ 0.3
y_true = np.ones(len(individual_meteor_scores))
y_pred = np.array([1 if s >= threshold else 0 for s in individual_meteor_scores])

accuracy  = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall    = recall_score(y_true, y_pred, zero_division=0)
f1        = f1_score(y_true, y_pred, zero_division=0)
mcc       = matthews_corrcoef(y_true, y_pred)
cm        = confusion_matrix(y_true, y_pred)

print("\n Derived Classification & Regression Metrics:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"MCC      : {mcc:.4f}")
print("\nConfusion Matrix:\n", cm)


summary = {
    "BLEU": bleu_score['bleu'],
    "METEOR": meteor_score['meteor'],
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "MCC": mcc,
}
pd.DataFrame([summary]).to_csv("evaluation_summary.csv", index=False)
print("\n Evaluation summary saved to evaluation_summary.csv")
