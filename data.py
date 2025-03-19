import os
import re
import json
import torch

from zhconv import convert
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

corpus_file = 'cleaned_corpus.txt'
model_dirname = 'tokenizer_chinese'

ROLE_MAP = {"system": 0, "user": 1, "assistant": 2}

def clean_text(text):
    # 定义正则表达式模式匹配对话块
    pattern = r'<\|start_header_id\|>(.*?)<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>'
    
    # 查找所有匹配项
    matches = re.findall(pattern, text, re.DOTALL)
    
    # 初始化结果列表和临时存储
    result = []
    last_role = last_content = None
    
    for role, content in matches:
        content = convert(content.strip(), 'zh-cn')
        
        if last_role == role:
            last_content += '\n\n' + content
        else:
            if last_role:
                result.append({"role": last_role, "content": last_content})
            last_role, last_content = role, content
    
    if last_role:
        result.append({"role": last_role, "content": last_content})
    return result

if not os.path.exists(corpus_file):
    
    dataset = load_dataset('neo-lin/chat_alpaca_chinese_llama_3.1', split='train')
    
    # 应用清洗并保存
    with open(corpus_file, "w", encoding="utf-8") as f:
        for example in dataset:
            cleaned = clean_text(example["text"])
            if cleaned:
                f.write(json.dumps(cleaned, ensure_ascii=False) + "\n")

if not os.path.exists(model_dirname):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    tokenizer.save_pretrained(model_dirname)

def create_tokenizer():
    return AutoTokenizer.from_pretrained(model_dirname)

class DialogueDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.data = []
        self.tokenizer = tokenizer
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        dialogue = self.data[idx]
        input_ids = []
        role_ids = []
        
        sep_token = self.tokenizer.special_tokens_map['sep_token']
        sep_id = self.tokenizer.convert_tokens_to_ids(sep_token)
        for item in dialogue:
            role, content = item['role'], item['content']
            tokens = self.tokenizer.encode(content, add_special_tokens=False)
            tokens.append(sep_id)
            input_ids.extend(tokens)
            role_ids.extend([ROLE_MAP[role]] * len(tokens))
        
        return {
            "input_ids": input_ids,
            "role_ids": role_ids
        }

def collate_batch(batch, tokenizer, max_len):
    input_batch = []
    role_batch = []
    
    for item in batch:
        input_batch.append(torch.LongTensor(item['input_ids'][:max_len]))
        role_batch.append(torch.LongTensor(item['role_ids'][:max_len]))
    
    pad_token = tokenizer.special_tokens_map['pad_token']
    pad_id = tokenizer.convert_tokens_to_ids(pad_token)
    
    input_batch = pad_sequence(input_batch, batch_first=True, padding_value=pad_id)
    role_batch = pad_sequence(role_batch, batch_first=True, padding_value=0)
    return input_batch, role_batch

def create_dataloader(tokenizer, batch_size, max_len=512, shuffle=False, drop_last=False):
    dataset = DialogueDataset(corpus_file, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
        collate_fn=lambda batch: collate_batch(batch, tokenizer, max_len))
