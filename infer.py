import os
import torch

from data import create_tokenizer
from ModelArgs import ModelArgs
from DeepSeek import DeepSeek

ROLE_MAP = {"system": 0, "user": 1, "assistant": 2}

def process_data(input_ids, role_ids, model, role, tokens, device='cpu'):
    input_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
    role_tensor = torch.LongTensor([ROLE_MAP[role]] * len(tokens)).unsqueeze(0).to(device)
    
    input_ids = torch.cat([input_ids, input_tensor], dim=-1)[:, -model.max_seq_len:]
    role_ids = torch.cat([role_ids, role_tensor], dim=-1)[:, -model.max_seq_len:]
    
    return input_ids, role_ids

def get_probs(model, input_ids, prev_pos, tokenizer, temperature=1.0, top_k=None):
    # input_ids = input_ids[:, -model.max_seq_len:]
    # role_ids = role_ids[:, -model.max_seq_len:]
    
    pad_token = tokenizer.special_tokens_map['pad_token']
    pad_id = tokenizer.convert_tokens_to_ids(pad_token)
    mask = model.generate_mask(input_ids, pad_id)
    
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            start_pos=prev_pos,
            mask=mask
        )
    
    # 应用温度缩放，防止温度过低导致数值不稳定（设置温度下限为1e-5）
    output = output[:, -1] / max(temperature, 1e-5)
    # 生成指数分布噪声并与概率相除
    output.div_(torch.empty_like(output).exponential_(1))
    # Top-k 过滤
    if top_k is not None and 0 < top_k <= output.size(-1):
        indices_to_remove = output < torch.topk(output, top_k)[0][..., -1, None]
        output[indices_to_remove] = float('-inf')
    return torch.softmax(output, dim=-1)

def sampling_decode(model, input_ids, role_ids, tokenizer, max_len=100, temperature=1.0, top_k=1):
    model.eval()
    
    sep_token = tokenizer.special_tokens_map['sep_token']
    sep_id = tokenizer.convert_tokens_to_ids(sep_token)
    assistant_id = torch.LongTensor([ROLE_MAP['assistant']]).unsqueeze(0).to(device)
    result = []
    
    input_len = input_ids.size(-1)
    prev_pos = 0  # 记录前一次处理的位置
    for cur_pos in range(input_len, input_len+max_len):
        probs = get_probs(model, input_ids[:, prev_pos:cur_pos], prev_pos, tokenizer, temperature, top_k)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        # role_ids = torch.cat([role_ids, assistant_id], dim=-1)
        result.append(next_token.item())
        prev_pos = cur_pos
        if result[-1] == sep_id:
            break
    if result[-1] != sep_id:
        result.append(sep_id)
    
    return result

if __name__ == '__main__':
    tokenizer = create_tokenizer()
    
    # 创建模型
    args = ModelArgs()
    args.vocab_size = tokenizer.vocab_size
    model = DeepSeek(args)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    ckpt_path = './checkpoints/checkpoint_best.pth'
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    input_ids = torch.LongTensor().unsqueeze(0).to(device)
    role_ids = torch.LongTensor().unsqueeze(0).to(device)
    
    sep_token = tokenizer.special_tokens_map['sep_token']
    sep_id = tokenizer.convert_tokens_to_ids(sep_token)
    
    while True:
        
        while True:
            try:
                text = input('>>> ').strip()
                print()
            except:
                print()
                exit()
            
            if text:
                break
        
        if text == '/exit':
            break
        
        if text == '/clear':
            input_ids = torch.LongTensor().unsqueeze(0).to(device)
            role_ids = torch.LongTensor().unsqueeze(0).to(device)
            print('历史已清除')
            print()
            continue
        
        tokens = tokenizer.encode(text, add_special_tokens=False)
        tokens.append(sep_id)
        
        input_ids, role_ids = process_data(input_ids, role_ids, model, 'user', tokens, device=device)
        
        predictions = sampling_decode(model, input_ids, role_ids, tokenizer, max_len=100, temperature=0.9, top_k=5)
        input_ids, role_ids = process_data(input_ids, role_ids, model, 'assistant', predictions, device=device)
        
        print(tokenizer.decode(predictions, skip_special_tokens=True).replace(" ", ""))
        print()
