import os
import torch

from data import create_tokenizer
from ModelArgs import ModelArgs
from Transformer import Transformer

def get_probs(model, input_ids, prev_pos, tokenizer, temperature=1.0):
    # input_ids = input_ids[:, -model.max_seq_len:]
    mask = model.generate_mask(input_ids, tokenizer.pad_token_id)
    
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            start_pos=prev_pos,
            mask=mask
        )
    
    if temperature > 0:
        # 应用温度缩放，防止温度过低导致数值不稳定（设置温度下限为1e-5）
        output = output[:, -1] / max(temperature, 1e-5)
        probs = torch.softmax(output, dim=-1)
        # 生成指数分布噪声并与概率相除
        return probs.div_(torch.empty_like(probs).exponential_(1))
    return output

def sampling_decode(model, input_ids, tokenizer, max_len=100, temperature=1.0):
    # 验证输入长度不超过模型限制
    input_len = input_ids.size(-1)
    if input_len > model.max_seq_len:
        print(f"输入长度超过模型最大长度限制（max_seq_len={model.max_seq_len}）")
        exit()
    max_len = min(max_len, model.max_seq_len - input_len)
    
    model.eval()
    
    result = []
    prev_pos = 0  # 记录前一次处理的位置
    for cur_pos in range(input_len, input_len+max_len):
        probs = get_probs(model, input_ids[:, prev_pos:cur_pos], prev_pos, tokenizer, temperature)
        next_token = probs.argmax(dim=-1).unsqueeze(-1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        result.append(next_token.item())
        prev_pos = cur_pos
        if next_token.item() == tokenizer.eos_token_id:
            break
    if not result or result[-1] != tokenizer.eos_token_id:
        result.append(tokenizer.eos_token_id)
    
    return result

if __name__ == '__main__':
    # 设置随机种子（保证可重复性）
    torch.manual_seed(0)
    
    tokenizer = create_tokenizer()
    
    # 创建模型
    args = ModelArgs()
    args.vocab_size = len(tokenizer.get_vocab())
    model = Transformer(args)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    ckpt_path = './checkpoints/checkpoint_best.pth'
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    default_messages = [
        {'role': 'system', 'content': '你是一个强大的助手。'}
    ]
    messages = default_messages[:]
    
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
            messages = default_messages[:]
            print('历史已清除')
            print()
            continue
        
        messages.append({'role': 'user', 'content': text})
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        input_ids = torch.LongTensor(prompt).unsqueeze(0).to(device)
        
        predictions = sampling_decode(model, input_ids, tokenizer, max_len=100, temperature=0.9)
        result = tokenizer.decode(predictions, skip_special_tokens=True)
        messages.append({'role': 'assistant', 'content': result})
        
        print(result)
        print()
