from transformers import BertJapaneseTokenizer
from torch.nn.functional import cosine_similarity
import torch
from torch.utils.data import Dataset

from transformers import TrainingArguments
from transformers import EarlyStoppingCallback
from transformers import Trainer

import os
import json
import random

from model import PersonaCLRModel

class PersonaCLRDataset(Dataset):
    def __init__(self, jsonfile_list, is_valid=False, single_reference=False, shuffle=True):
        self.features = []
        for jsonfile in jsonfile_list:
            with open(jsonfile) as f:
                for l in f:
                    jdata = json.loads(l)

                    
                    target = jdata["target"]
                    
                    if single_reference:
                        reference = jdata["reference"][0]
                    else:
                        reference = "[SEP]".join(jdata["reference"])
                    
                    if "labels" in jdata:
                        label = jdata["labels"]
                        self.features.append({'target': target, "reference": reference, "labels": label, "is_valid": is_valid})
                    else:
                        self.features.append({'target': target, "reference": reference, "is_valid": is_valid})
        if shuffle:
            random.shuffle(self.features)
            
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]
    
class PersonaCLRCollator():
    def __init__(self, bert_tokenizer, max_length=512):
        self.bert_tokenizer = bert_tokenizer
        self.max_length = max_length
    
    def __call__(self, examples):
        target = []
        reference = []
        is_valid = examples[0]["is_valid"] 
        # label = []
        for e in examples:
            target.append(e["target"])
            reference.append(e["reference"])

        data1 = self.bert_tokenizer.batch_encode_plus(target, padding='longest', truncation=True, max_length=self.max_length, return_tensors='pt')
        data2 = self.bert_tokenizer.batch_encode_plus(reference, padding='longest', truncation=True, max_length=self.max_length, return_tensors='pt')

        
        labels = torch.arange(len(examples), dtype=torch.long)
        return {'target_dict': data1, "reference_dict": data2, "labels":labels, "is_valid":is_valid}

       

if __name__ == '__main__':
    bert_tokenizer = BertJapaneseTokenizer.from_pretrained("nlp-waseda/roberta-large-japanese-with-auto-jumanpp")
    model = PersonaCLRModel('nlp-waseda/roberta-large-japanese-with-auto-jumanpp', tau=0.05)

    train_dataset = PersonaCLRDataset(["data/processed/train.jsonl", "data/processed/train_random.jsonl"])
    valid_dataset = PersonaCLRDataset(["data/processed/valid.jsonl", "data/processed/valid_random.jsonl"], is_valid=True)

    collator_fn = PersonaCLRCollator(bert_tokenizer)
    output_dir = 'output/'

    os.makedirs(output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir = 'True',
        eval_strategy='steps',
        logging_strategy='steps',
        logging_steps=100,
        save_steps=100,
        learning_rate=1e-5,
        metric_for_best_model='loss',
        warmup_steps=100,
        load_best_model_at_end=True,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        remove_unused_columns=False,
        tf32=True,
        optim="adafactor",
    )


    trainer = Trainer(
        model=model,
        tokenizer=bert_tokenizer,
        data_collator=collator_fn,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )
    
    
    trainer.train()
    trainer.save_model(output_dir)