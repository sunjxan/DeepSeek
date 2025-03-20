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
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")
    tokenizer.chat_template = '''{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}
{% set ns = namespace(system_prompt='', is_first_sp=true) %}
{% for message in messages %}
    {% if message['role'] == 'system' %}
        {% if ns.is_first_sp %}
            {% set ns.system_prompt = ns.system_prompt + message['content'] %}
            {% set ns.is_first_sp = false %}
        {% else %}
            {% set ns.system_prompt = ns.system_prompt + '\\n' + message['content'] %}
        {% endif %}
    {% endif %}
{% endfor %}{{ bos_token }}{{ ns.system_prompt }}{% for message in messages %}
    {% if message['role'] == 'user' %}{{'<｜User｜>' + message['content']}}{% endif %}
    {% if message['role'] == 'assistant' %}{{'<｜Assistant｜>' + message['content'] + eos_token}}{% endif %}
{% endfor %}
{% if add_generation_prompt %}{{'<｜Assistant｜>'}}{% endif %}'''
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
        input_ids = self.tokenizer.apply_chat_template(self.data[idx])
        role_ids = [0] * len(input_ids)
        
        user_id = self.tokenizer.convert_tokens_to_ids("<｜User｜>")
        assistant_id = self.tokenizer.convert_tokens_to_ids("<｜Assistant｜>")
        
        last_pos = None
        last_token = None
        for i, v in enumerate(input_ids):
            if v in [user_id, assistant_id]:
                if last_pos is not None:
                    role = 2 if last_token == assistant_id else 1
                    for j in range(last_pos+1, i):
                        role_ids[j] = role
                last_pos = i
                last_token = v
        if last_pos is not None:
            role = 2 if last_token == assistant_id else 1
            for j in range(last_pos+1, len(role_ids)):
                role_ids[j] = role
        
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
        
    input_batch = pad_sequence(input_batch, batch_first=True, padding_value=tokenizer.pad_token_id)
    role_batch = pad_sequence(role_batch, batch_first=True, padding_value=0)
    return input_batch, role_batch

def create_dataloader(tokenizer, batch_size, max_len=512, shuffle=False, drop_last=False):
    dataset = DialogueDataset(corpus_file, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
        collate_fn=lambda batch: collate_batch(batch, tokenizer, max_len))
