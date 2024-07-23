import json
import glob
from rank_bm25 import BM25Okapi
import MeCab
import ipadic
import random
import sys
import os
from tqdm import tqdm
import util

random.seed = 1
MIN_UTTR_LENGTH = 5


# Number of utterance pool
SEARCH_UTTR_NUM = 20
# Characters with fewer lines than this number are not used
MIN_UTTR_NUM = 10

def get_train_uttrs():
    train_uttr_list = []
    for filename in glob.glob(f"data/NaroU/train/*.json"):
        with open(filename) as f:
            jdata = json.load(f)
            
            for name, uttrs in jdata.items():
                for u in uttrs:
                    u = util.normalize(u)
                    if util.word_count(u) >= MIN_UTTR_LENGTH and u not in train_uttr_list:
                        train_uttr_list.append(u)
    # print("train uttr num", len(train_uttr_list))
    return train_uttr_list
                        
def preprocess_train_valid():
    train_uttr_list = get_train_uttrs()

    for is_random in [True, False]:
        for datatype in ["train", "valid"]:
            character_uttr_dict = {}
            character_id = 0 

            series_ids_list = []
            data_all_uttrs = []
            print("Loading NaroU", end=" ")
            for filename in glob.glob(f"data/NaroU/{datatype}/*.json"):
                with open(filename) as f:
                    jdata = json.load(f)
                    series_ids = []
                    
                    for _, uttrs in jdata.items():
                        tmp_uttrs = []
                        for u in uttrs:
                            u = util.normalize(u)
                            if util.word_count(u) >= MIN_UTTR_LENGTH and u not in tmp_uttrs:
                                tmp_uttrs.append(u)
                        if len(tmp_uttrs) >= MIN_UTTR_NUM:
                            tmp_uttrs = list(set(tmp_uttrs))
                            data_all_uttrs += tmp_uttrs
                            character_uttr_dict[character_id] = tmp_uttrs
                            series_ids.append(character_id)
                            character_id += 1
                    series_ids_list.append(series_ids)
            print("... finished")

            tagger = MeCab.Tagger(f'-O wakati {ipadic.MECAB_ARGS}')

            data_list = []
            
            if is_random is False:
                all_uttrs = data_all_uttrs + train_uttr_list
                all_uttrs = list(set(all_uttrs))
                tokenized_corpus = [util.parse(ui).split() for ui in all_uttrs]
                bm25 = BM25Okapi(tokenized_corpus)
            
            print(len(data_all_uttrs))
            for k, v in tqdm(character_uttr_dict.items()):
                for vi in v:
                    # 正解データを作成
                    target = vi
                    query = tagger.parse(target).split()

                    if is_random:
                        if len(v) > SEARCH_UTTR_NUM:
                            related_uttrs = random.sample(v, SEARCH_UTTR_NUM+1)
                        else:
                            related_uttrs = v
                        related_uttrs = list(set(related_uttrs))
                        if target in related_uttrs:
                            related_uttrs.remove(target)
                        related_uttrs = related_uttrs[:SEARCH_UTTR_NUM]
                        
                        
                        if len(related_uttrs) >= MIN_UTTR_NUM :
                            data_list.append({"label":k, "target": target, "reference": related_uttrs})
                    else:
                        nearest_uttrs = []
                    
                        topn_uttrs = bm25.get_top_n(query, all_uttrs, n=len(tokenized_corpus))
                        

                        
                        for t in topn_uttrs:
                            if t in v and target != t and t not in nearest_uttrs: #評価対象発話とは異なり，保存済み発話とも重複していない場合
                                nearest_uttrs.append(t)
                        if len(nearest_uttrs) >= MIN_UTTR_NUM :
                            data_list.append({"label":k, "target": target, "reference": nearest_uttrs[:SEARCH_UTTR_NUM]})

            print(datatype, "character size", len(character_uttr_dict), "data size", len(data_list))
            
            if is_random:
                write_filename = f"data/processed/{datatype}_random.jsonl"
            else:
                write_filename = f"data/processed/{datatype}.jsonl"
            with open(write_filename, "w") as w:
                for l in data_list:
                    w.write(json.dumps(l, ensure_ascii=False) + "\n")

def preprocess_test():
    train_uttr_list = get_train_uttrs()
    data_tuples = util.get_data_tuples()
    for narou_file, character_name, en_name in data_tuples:
        print(narou_file)
        character_uttrs = []
        with open(narou_file) as f:
            jdata = json.load(f)
            
            for u in jdata[character_name]:
                u = util.normalize(u)
                if util.word_count(u) >= MIN_UTTR_LENGTH and u not in character_uttrs:
                    character_uttrs.append(u)
   
        all_uttrs = character_uttrs + train_uttr_list
        tokenized_corpus = [util.parse(ui).split() for ui in all_uttrs]
        bm25 = BM25Okapi(tokenized_corpus)

        data_list = []
        with open(f"data/evaluation/{en_name}.json") as f:
            jdata = json.load(f)
            for k, v in jdata.items():
                target = k.split("\t")[1]
                target = util.normalize(target)
                query = util.parse(target).split()
                

                related_uttrs = []
                
                tmp_uttrs = bm25.get_top_n(query, all_uttrs, n=len(tokenized_corpus))
                for t in tmp_uttrs:
                    if t in character_uttrs and t not in related_uttrs:
                        related_uttrs.append(t)
                    if len(related_uttrs) >= SEARCH_UTTR_NUM:
                            break
                data_list.append({"labels":v, "target": target, "reference": related_uttrs})
                
        with open(f"data/processed/test_{en_name}.jsonl", "w") as w:
            for l in data_list:
                w.write(json.dumps(l, ensure_ascii=False) + "\n")
preprocess_test()
# preprocess_train_valid()
    # train character_size 507 uttr len 21213
    # valid character_size 57 uttr len 2801