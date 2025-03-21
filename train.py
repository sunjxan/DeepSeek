import os
import time
import torch
import torch.nn as nn

from torch.functional import F
from torch.utils.tensorboard import SummaryWriter

from data import create_tokenizer, create_dataloader
from Transformer import Transformer
from ModelArgs import ModelArgs

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, calc_accuracy, optimizer, scheduler=None, config=None):
        """
        初始化训练器
        Args:
            model: 待训练的模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            criterion: 损失函数
            calc_accuracy: 计算准确率函数
            optimizer: 优化器
            scheduler: 学习率调度器 (可选)
            config: 配置字典，包含以下参数：
                device: 使用的设备 (cuda/cpu)
                epochs: 训练轮数
                save_dir: 模型保存路径
                log_dir: 日志保存路径
                checkpoint: 预训练权重路径 (可选)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.calc_accuracy = calc_accuracy
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        
        # 初始化设置
        self.device = config['device']
        self.epochs = config['epochs']
        self.save_dir = config['save_dir']
        self.log_dir = config['log_dir']
        self.print_interval_steps = config['print_interval_steps']
        self.save_interval_epochs = config['save_interval_epochs']
        self.best_val_loss = float('inf')
        
        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # TensorBoard Writer
        self.writer = SummaryWriter(self.log_dir)
        
        # 设备设置
        self.model.to(self.device)
        
        # 加载预训练权重
        if 'checkpoint' in config and config['checkpoint'] and os.path.exists(config['checkpoint']):
            self.load_checkpoint(config['checkpoint'])
        
        # 保存计算图
        self._save_computation_graph()
    
    def _save_computation_graph(self):
        """保存模型计算图到TensorBoard"""
        input_ids, role_ids = next(iter(self.train_loader))
        input_shape = input_ids.shape[1:]
        dummy_input = torch.zeros(1, *input_shape).long().to(self.device)
        start_pos = torch.zeros(1).int()
        dummy_mask = torch.zeros(1, *input_shape, *input_shape).bool().to(self.device)
        self.writer.add_graph(self.model, (dummy_input, start_pos, dummy_mask))
    
    def train_epoch(self, epoch):
        """训练单个epoch"""
        self.model.train()
        total_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (input_ids, role_ids) in enumerate(self.train_loader):
            iter_start_time = time.time()
            
            input_ids = input_ids.to(self.device)
            role_ids = role_ids.to(self.device)
            
            # 生成掩码
            mask = model.generate_mask(input_ids, self.config['pad_id'])
            
            # 梯度清零
            self.optimizer.zero_grad()
            
            # 前向传播
            output = model(
                input_ids=input_ids[:, :-1],  # 解码器输入去尾
                mask=mask[:, :-1, :-1]
            )
            
            # 计算损失
            loss = self.criterion(
                output.contiguous().view(-1, output.size(-1)),
                input_ids[:, 1:].contiguous().view(-1),  # 目标去头
                role_ids[:, 1:].contiguous().view(-1)
            )
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 参数更新
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 记录单步时间
            iter_time = time.time() - iter_start_time
            if batch_idx % self.print_interval_steps == 0:
                print(f'Train Epoch: {epoch} [{batch_idx+1}/{len(self.train_loader)}] Loss: {loss.item():.4f} Time: {iter_time:.3f}s')
                
            # TensorBoard记录
            global_step = (epoch - 1) * len(self.train_loader) + batch_idx + 1
            self.writer.add_scalar('train/step_loss', loss.item(), global_step)
            self.writer.add_scalar('train/step_time', iter_time, global_step)
        
        avg_loss = total_loss / len(self.train_loader)
        epoch_time = time.time() - start_time
        
        # 记录学习率
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('train/lr', current_lr, epoch)
        self.writer.add_scalar('train/epoch_loss', avg_loss, epoch)
        self.writer.add_scalar('train/epoch_time', epoch_time, epoch)
        
        return avg_loss, epoch_time
    
    def validate(self, epoch):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        accuracy_correct = accuracy_total = 0
        start_time = time.time()
        
        for input_ids, role_ids in self.val_loader:
            input_ids = input_ids.to(self.device)
            role_ids = role_ids.to(self.device)
            
            # 生成掩码
            mask = model.generate_mask(input_ids, self.config['pad_id'])
            
            # 前向传播
            with torch.no_grad():
                output = model(
                    input_ids=input_ids[:, :-1],  # 解码器输入去尾
                    mask=mask[:, :-1, :-1]
                )
            
            # 计算损失
            loss = self.criterion(
                output.contiguous().view(-1, output.size(-1)),
                input_ids[:, 1:].contiguous().view(-1),  # 目标去头
                role_ids[:, 1:].contiguous().view(-1)
            )
            
            total_loss += loss.item()
            
            correct, total = self.calc_accuracy(
                output.contiguous().view(-1, output.size(-1)),
                input_ids[:, 1:].contiguous().view(-1),  # 目标去头
                role_ids[:, 1:].contiguous().view(-1)
            )
            accuracy_correct += correct
            accuracy_total += total
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_correct / accuracy_total if accuracy_total else 0
        epoch_time = time.time() - start_time
        
        # TensorBoard记录
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        self.writer.add_scalar('val/accuracy', accuracy, epoch)
        self.writer.add_scalar('val/time', epoch_time, epoch)
        
        # 保存最佳模型
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint('checkpoint_best.pth')
            torch.save(model, 'best_model.pth')
        
        return avg_loss, accuracy, epoch_time
    
    def train(self):
        """完整训练流程"""
        for epoch in range(1, self.epochs + 1):
            train_loss, train_time = self.train_epoch(epoch)
            val_loss, accuracy, val_time = self.validate(epoch)
            
            # 更新学习率
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 打印日志
            print(f'\nEpoch: {epoch}/{self.epochs} Train Loss: {train_loss:.4f} Time: {train_time:.3f}s | '
                  f'Val Loss: {val_loss:.4f} Accuracy: {100*accuracy:.2f}% Time: {val_time:.3f}s\n')
            
            # 定期保存模型
            if (epoch - 1) % self.save_interval_epochs == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
        
        self.writer.close()
    
    def save_checkpoint(self, filename):
        """保存模型检查点"""
        checkpoint = {
            'epoch': self.epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss
        }
        torch.save(checkpoint, os.path.join(self.save_dir, filename))
    
    def load_checkpoint(self, checkpoint_path):
        """加载模型检查点"""
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        print(f'Loaded checkpoint from {checkpoint_path}')

# 示例用法
if __name__ == '__main__':
    # 初始化模型和数据加载器
    tokenizer = create_tokenizer()
    
    # 创建模型
    args = ModelArgs()
    args.vocab_size = len(tokenizer.get_vocab())
    model = Transformer(args)
    
    # 初始化参数
    model.init_parameters()
    
    # 只计算机器人回复部分的损失（用户发言设为ignore_index）
    def masked_loss(logits, labels, role_ids):
        mask = (role_ids == 2) & (labels != tokenizer.pad_token_id)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction='none'
        )
        return (loss * mask.view(-1).float()).mean()
    
    def calc_accuracy(logits, labels, role_ids):
        preds = torch.argmax(logits, dim=-1)
        mask = (role_ids == 2) & (labels != tokenizer.pad_token_id)
        correct = (preds[mask] == labels[mask]).sum().item()
        return correct, mask.sum().item()
    
    # 定义优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.01
    )
    
    # 定义学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=0.95
    )
    
    train_loader = create_dataloader(tokenizer, batch_size=10, max_len=model.max_seq_len, shuffle=True, drop_last=True)
    val_loader = create_dataloader(tokenizer, batch_size=10, max_len=model.max_seq_len)
    
    # 配置参数
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'epochs': 20,
        'save_dir': './checkpoints',
        'log_dir': './logs',
        'checkpoint': './checkpoints/checkpoint_best.pth',  # 可以指定预训练权重路径
        'print_interval_steps': 10,
        'save_interval_epochs': 5,
        'pad_id': tokenizer.pad_token_id
    }
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=masked_loss,
        calc_accuracy=calc_accuracy,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config
    )
    
    # 开始训练
    trainer.train()
