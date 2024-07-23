from transformers import BertJapaneseTokenizer, BertModel, PreTrainedModel, BertPreTrainedModel, BertForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments
from transformers import EarlyStoppingCallback
from transformers import Trainer
import torch
import math
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr
import glob
import json
import os
import sys
import util
from sklearn.metrics import precision_recall_curve, auc
from model import PersonaCLRModel
from train import PersonaCLRDataset
import argparse 
from safetensors.torch import load_model, save_model

eval_files = [
    "rimuru_test.jsonl",
    "veldora_test.jsonl",
    "rudeus_test.jsonl",
    "roxy_test.jsonl",
    "sylphy_test.jsonl",
    "paul_test.jsonl",
    "zenith_test.jsonl",
    "subaru_test.jsonl",
    "myne_test.jsonl",
    "tuuli_test.jsonl",
    "effa_test.jsonl",
    "leon_test.jsonl",
    "olivia_test.jsonl",
    "angelica_test.jsonl",
    "luxion_test.jsonl",
    "catarina_test.jsonl",
    "keith_test.jsonl",
]

class ConstractiveEvalCollator():
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, examples):
        target = []
        reference = []
        labels = []
        for e in examples:
            target.append(e["target"])
            reference.append(e["reference"])
            labels.append(e["labels"])
        data1 = self.tokenizer.batch_encode_plus(target, padding='longest', truncation=True, max_length=self.max_length, return_tensors='pt')
        data2 = self.tokenizer.batch_encode_plus(reference, padding='longest', truncation=True, max_length=self.max_length, return_tensors='pt')
        labels = torch.tensor(labels, dtype=torch.float)

        return {'target_dict': data1, "reference_dict": data2, "labels":labels}


tokenizer = AutoTokenizer.from_pretrained("nlp-waseda/roberta-large-japanese-with-auto-jumanpp")
bert_model_name_or_path = "nlp-waseda/roberta-large-japanese-with-auto-jumanpp"


parser = argparse.ArgumentParser()
parser.add_argument('--warmup', type=int, default=100)  
parser.add_argument('--lr', type=float, default=1e-5)  
parser.add_argument('--pooling', type=str, default="cls")  
parser.add_argument('--device', type=str, default="cuda")  
args = parser.parse_args()


    

result_dict = {}
data_tuples = util.get_data_tuples()


checkpoint = "model/model.safetensors"
test_dir = "data/evaluation/"

model = PersonaCLRModel(bert_model_name_or_path)
load_model(model, checkpoint)
# model.load_state_dict(torch.load(checkpoint))

model.to(args.device)
for _, _, en_name in data_tuples:
    valid_dataset = PersonaCLRDataset([f"data/processed/test_{en_name}.jsonl"])
    collator_fn = ConstractiveEvalCollator(tokenizer)
    loader = DataLoader(valid_dataset, collate_fn=collator_fn, batch_size=1, shuffle=False)

    est_list = []
    act_list = []
    all_one = []
    with torch.no_grad():
        for batch in loader:
            batch["target_dict"].to(args.device)
            batch["reference_dict"].to(args.device)
            batch["labels"] = batch["labels"].to(args.device)
            out = model.predict(**batch)
            for est, act in zip(out, batch["labels"]):
                est_list.append(est.item())
                act_list.append(act.item())
                all_one.append(1)
    correlation, pvalue = spearmanr(est_list, act_list)
    
    act_labels = []
    est_probs = []
    for e, a in zip(est_list, act_list):
        if e < 0.0:
            est_probs.append(1.0)
        else:
            est_probs.append(1.0 - e)
        if a < 2:
            act_labels.append(1)
        else:
            act_labels.append(0)
    precision, recall, thresholds = precision_recall_curve(act_labels, est_probs)
    pr_auc = auc(recall, precision)

    
    if en_name not in result_dict:
        result_dict[en_name] = []
    result_dict[en_name].append({"correlation":correlation, "pvalue":pvalue, "pr_auc":pr_auc})
    print(en_name, "correlation:", correlation, "p-value:", pvalue, "AUPR", pr_auc) 


