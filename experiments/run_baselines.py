"""
基线模型对比实验
在同一数据集、同一划分、同一评估协议下重跑以下基线：
1. BERT (bert-base-chinese)
2. RoBERTa-wwm-ext (hfl/chinese-roberta-wwm-ext)
3. TextCNN
4. BiLSTM-Attention

支持断点续跑
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from collections import Counter
from tqdm import tqdm
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    AutoTokenizer, AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.seeding import set_seed, get_device
from utils.data_loader import (
    load_raw_data, split_by_source, augment_training_only, build_vocab
)
from utils.metrics import compute_all_metrics


# ==================== 配置 ====================
BASE_PATH = r"C:\Users\DF-Lenovo\Desktop\数据集"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASELINE_SEEDS = [42, 2024]  # 基线用2个种子，节省时间

CONFIG = {
    'num_classes': 7,
    'max_length': 200,
    'batch_size': 16,
    'epochs': 60,
    'patience': 15,
    'lr': 3e-4,
    'weight_decay': 1e-4,
    'warmup_steps': 300,
    'augment_target': 800,
}


# ==================== 数据集 ====================
class TextClassificationDataset(Dataset):
    """通用文本分类数据集（用于TextCNN和BiLSTM-Attention）"""
    def __init__(self, texts, labels, vocab, max_len=200):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx][:self.max_len]
        ids = [self.vocab.get(c, 1) for c in text]
        ids = ids + [0] * (self.max_len - len(ids))
        mask = [1 if i != 0 else 0 for i in ids]
        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
        }


class TransformerDataset(Dataset):
    """BERT/RoBERTa 数据集"""
    def __init__(self, texts, labels, tokenizer, max_len=200):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ==================== TextCNN 模型 ====================
class TextCNN(nn.Module):
    """经典TextCNN用于文本分类"""
    def __init__(self, vocab_size, embed_dim=300, num_classes=7,
                 filter_sizes=(2, 3, 4, 5), num_filters=128, dropout=0.5):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim))
            for k in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
    
    def forward(self, input_ids, attention_mask=None):
        # (B, L) -> (B, L, E)
        x = self.embedding(input_ids)
        # (B, 1, L, E)
        x = x.unsqueeze(1)
        
        conv_outputs = []
        for conv in self.convs:
            # (B, num_filters, L - k + 1, 1)
            c = torch.relu(conv(x)).squeeze(3)
            # (B, num_filters)
            p = torch.max_pool1d(c, c.size(2)).squeeze(2)
            conv_outputs.append(p)
        
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits, None


# ==================== BiLSTM-Attention 模型 ====================
class BiLSTMAttention(nn.Module):
    """经典BiLSTM + Attention用于文本分类"""
    def __init__(self, vocab_size, embed_dim=300, hidden_dim=384,
                 num_classes=7, num_layers=3, dropout=0.4, attention_heads=8):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim // 2,
            num_layers=num_layers, batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.lstm_norm = nn.LayerNorm(hidden_dim)
        
        self.attention = nn.MultiheadAttention(
            hidden_dim, attention_heads,
            dropout=0.1, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, 192),
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(192, num_classes)
        )
    
    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_norm(lstm_out)
        
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None
        
        attn_out, _ = self.attention(
            lstm_out, lstm_out, lstm_out,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        attn_out = self.attention_norm(lstm_out + attn_out)
        
        max_pool = torch.max(attn_out, dim=1)[0]
        avg_pool = torch.mean(attn_out, dim=1)
        last = attn_out[:, -1, :]
        
        features = torch.cat([max_pool, avg_pool, last], dim=1)
        logits = self.classifier(features)
        return logits, None


# ==================== 通用训练函数（用于TextCNN/BiLSTM） ====================
def train_pytorch_model(
    model, train_loader, val_loader, test_loader,
    model_name, seed, config=CONFIG
):
    """训练 PyTorch 原生模型（TextCNN / BiLSTM-Attention）"""
    device = get_device()
    set_seed(seed)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    total_steps = len(train_loader) * config['epochs']
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=total_steps
    )
    
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    best_state = None
    patience_cnt = 0
    
    for epoch in range(config['epochs']):
        model.train()
        for batch in train_loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            logits, _ = model(ids, mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        # 验证
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                logits, _ = model(
                    batch['input_ids'].to(device),
                    batch['attention_mask'].to(device)
                )
                val_preds.extend(logits.argmax(1).cpu().numpy())
                val_true.extend(batch['label'].numpy())
        
        val_acc = accuracy_score(val_true, val_preds)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= config['patience']:
                break
    
    # 测试
    model.load_state_dict(best_state)
    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            logits, _ = model(
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device)
            )
            test_preds.extend(logits.argmax(1).cpu().numpy())
            test_true.extend(batch['label'].numpy())
    
    return compute_all_metrics(test_true, test_preds)


# ==================== BERT / RoBERTa 训练函数 ====================
def train_transformer_model(
    model_name, pretrained_path, train_texts, train_labels,
    val_texts, val_labels, test_texts, test_labels,
    seed, config=CONFIG
):
    """训练 BERT 或 RoBERTa"""
    device = get_device()
    set_seed(seed)
    
    print(f"  Loading {model_name} from {pretrained_path}...")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_path, num_labels=config['num_classes']
    ).to(device)
    
    # 数据集
    train_ds = TransformerDataset(train_texts, train_labels, tokenizer, config['max_length'])
    val_ds = TransformerDataset(val_texts, val_labels, tokenizer, config['max_length'])
    test_ds = TransformerDataset(test_texts, test_labels, tokenizer, config['max_length'])
    
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=2e-5,  # BERT/RoBERTa 的标准学习率
        weight_decay=config['weight_decay']
    )
    
    total_steps = len(train_loader) * config['epochs']
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=total_steps
    )
    
    best_val_acc = 0
    best_state = None
    patience_cnt = 0
    
    for epoch in range(config['epochs']):
        model.train()
        for batch in train_loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        # 验证
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device)
                )
                val_preds.extend(outputs.logits.argmax(1).cpu().numpy())
                val_true.extend(batch['label'].numpy())
        
        val_acc = accuracy_score(val_true, val_preds)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= config['patience']:
                break
    
    # 测试
    model.load_state_dict(best_state)
    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device)
            )
            test_preds.extend(outputs.logits.argmax(1).cpu().numpy())
            test_true.extend(batch['label'].numpy())
    
    return compute_all_metrics(test_true, test_preds)


# ==================== 主函数 ====================
def main():
    print("="*60)
    print("Baseline Comparison: Re-train on identical dataset")
    print(f"Baselines: BERT, RoBERTa-wwm-ext, TextCNN, BiLSTM-Attention")
    print(f"Seeds: {BASELINE_SEEDS}")
    print("="*60)
    
    device = get_device()
    all_baseline_results = {}
    
    for seed in BASELINE_SEEDS:
        print(f"\n{'='*60}")
        print(f"Seed = {seed}")
        print(f"{'='*60}")
        
        # 加载数据（与 DHNN 主实验完全相同的划分方式）
        set_seed(seed)
        texts, labels, sources = load_raw_data(BASE_PATH)
        train_t, train_l, val_t, val_l, test_t, test_l = split_by_source(
            texts, labels, sources, seed=seed
        )
        
        # 仅在训练集内增强
        train_t, train_l = augment_training_only(
            train_t, train_l, target=CONFIG['augment_target'], seed=seed
        )
        
        # 构建词表（仅用训练集）
        vocab = build_vocab(train_t, 60000)
        
        # ========================================
        # 基线1：TextCNN
        # ========================================
        cnn_result_path = os.path.join(OUTPUT_DIR, f'baseline_textcnn_seed_{seed}.json')
        if os.path.exists(cnn_result_path):
            print(f"\n[Skip] TextCNN seed {seed} already done.")
            with open(cnn_result_path, 'r', encoding='utf-8') as f:
                all_baseline_results.setdefault('textcnn', []).append(json.load(f))
        else:
            print(f"\n[Run] TextCNN | seed {seed}")
            try:
                train_ds = TextClassificationDataset(train_t, train_l, vocab)
                val_ds = TextClassificationDataset(val_t, val_l, vocab)
                test_ds = TextClassificationDataset(test_t, test_l, vocab)
                
                train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
                val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
                test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
                
                model = TextCNN(len(vocab), embed_dim=300, num_classes=7).to(device)
                result = train_pytorch_model(
                    model, train_loader, val_loader, test_loader,
                    'TextCNN', seed
                )
                result['model'] = 'TextCNN'
                result['seed'] = seed
                
                with open(cnn_result_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                all_baseline_results.setdefault('textcnn', []).append(result)
                print(f"  TextCNN Test acc: {result['accuracy']:.4f}")
            except Exception as e:
                print(f"  TextCNN failed: {e}")
                import traceback
                traceback.print_exc()
        
        # ========================================
        # 基线2：BiLSTM-Attention
        # ========================================
        blstm_result_path = os.path.join(OUTPUT_DIR, f'baseline_bilstm_attn_seed_{seed}.json')
        if os.path.exists(blstm_result_path):
            print(f"\n[Skip] BiLSTM-Attention seed {seed} already done.")
            with open(blstm_result_path, 'r', encoding='utf-8') as f:
                all_baseline_results.setdefault('bilstm_attn', []).append(json.load(f))
        else:
            print(f"\n[Run] BiLSTM-Attention | seed {seed}")
            try:
                train_ds = TextClassificationDataset(train_t, train_l, vocab)
                val_ds = TextClassificationDataset(val_t, val_l, vocab)
                test_ds = TextClassificationDataset(test_t, test_l, vocab)
                
                train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
                val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
                test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
                
                model = BiLSTMAttention(
                    len(vocab), embed_dim=300, hidden_dim=384,
                    num_classes=7, num_layers=3, attention_heads=8
                ).to(device)
                result = train_pytorch_model(
                    model, train_loader, val_loader, test_loader,
                    'BiLSTM-Attention', seed
                )
                result['model'] = 'BiLSTM-Attention'
                result['seed'] = seed
                
                with open(blstm_result_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                all_baseline_results.setdefault('bilstm_attn', []).append(result)
                print(f"  BiLSTM-Attention Test acc: {result['accuracy']:.4f}")
            except Exception as e:
                print(f"  BiLSTM-Attention failed: {e}")
                import traceback
                traceback.print_exc()
        
        # ========================================
        # 基线3：BERT
        # ========================================
        bert_result_path = os.path.join(OUTPUT_DIR, f'baseline_bert_seed_{seed}.json')
        if os.path.exists(bert_result_path):
            print(f"\n[Skip] BERT seed {seed} already done.")
            with open(bert_result_path, 'r', encoding='utf-8') as f:
                all_baseline_results.setdefault('bert', []).append(json.load(f))
        else:
            print(f"\n[Run] BERT | seed {seed}")
            try:
                result = train_transformer_model(
                    'BERT', 'bert-base-chinese',
                    train_t, train_l, val_t, val_l, test_t, test_l,
                    seed
                )
                result['model'] = 'BERT'
                result['seed'] = seed
                
                with open(bert_result_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                all_baseline_results.setdefault('bert', []).append(result)
                print(f"  BERT Test acc: {result['accuracy']:.4f}")
            except Exception as e:
                print(f"  BERT failed: {e}")
                import traceback
                traceback.print_exc()
        
        # ========================================
        # 基线4：RoBERTa-wwm-ext
        # ========================================
        roberta_result_path = os.path.join(OUTPUT_DIR, f'baseline_roberta_seed_{seed}.json')
        if os.path.exists(roberta_result_path):
            print(f"\n[Skip] RoBERTa-wwm-ext seed {seed} already done.")
            with open(roberta_result_path, 'r', encoding='utf-8') as f:
                all_baseline_results.setdefault('roberta', []).append(json.load(f))
        else:
            print(f"\n[Run] RoBERTa-wwm-ext | seed {seed}")
            try:
                result = train_transformer_model(
                    'RoBERTa-wwm-ext', 'hfl/chinese-roberta-wwm-ext',
                    train_t, train_l, val_t, val_l, test_t, test_l,
                    seed
                )
                result['model'] = 'RoBERTa-wwm-ext'
                result['seed'] = seed
                
                with open(roberta_result_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                all_baseline_results.setdefault('roberta', []).append(result)
                print(f"  RoBERTa Test acc: {result['accuracy']:.4f}")
            except Exception as e:
                print(f"  RoBERTa failed: {e}")
                import traceback
                traceback.print_exc()
    
    # ========================================
    # 汇总
    # ========================================
    print("\n" + "="*60)
    print("BASELINE SUMMARY")
    print("="*60)
    print(f"{'Model':<25} {'Accuracy':<20} {'F1 Weighted':<20}")
    print("-" * 65)
    
    summary = {}
    for model_key, results in all_baseline_results.items():
        if not results:
            continue
        accs = [r['accuracy'] for r in results]
        f1ws = [r['f1_weighted'] for r in results]
        
        model_name = results[0].get('model', model_key)
        summary[model_key] = {
            'model': model_name,
            'accuracy_mean': float(np.mean(accs)),
            'accuracy_std': float(np.std(accs)),
            'f1_weighted_mean': float(np.mean(f1ws)),
            'f1_weighted_std': float(np.std(f1ws)),
            'individual_results': results,
        }
        
        print(f"{model_name:<25} "
              f"{np.mean(accs)*100:.2f} ± {np.std(accs)*100:.2f}      "
              f"{np.mean(f1ws)*100:.2f} ± {np.std(f1ws)*100:.2f}")
    
    # 保存
    summary_path = os.path.join(OUTPUT_DIR, 'baselines_results.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] {summary_path}")


if __name__ == '__main__':
    main()