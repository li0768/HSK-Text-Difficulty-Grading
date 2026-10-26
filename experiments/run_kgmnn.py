"""
KGMNN 同划分重训（仅 seed 123，作为基线）
- 与主模型相同的 load_raw_data / split_by_source / augment_training_only
- 同一评估脚本 compute_all_metrics
- 保存 test_probs / true_labels，可直接与所提模型同划分比较
- JSON 原子写入 + 回读校验
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
from transformers import get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.seeding import set_seed, get_device
from utils.data_loader import (
    load_raw_data, split_by_source, augment_training_only, build_vocab
)
from utils.metrics import compute_all_metrics


BASE_PATH = r"C:\Users\DF-Lenovo\Desktop\数据集"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 基线只跑主 seed
SEEDS = [123]

CONFIG = {
    'vocab_size': 60000,
    'embed_dim': 200,
    'hidden_dim': 256,
    'num_layers': 2,
    'num_classes': 7,
    'conv_channels': 96,
    'attention_heads': 8,
    'max_length': 200,
    'dropout': 0.5,
    'batch_size': 16,
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'epochs': 80,
    'patience': 20,
    'augment_target': 800,
    'warmup_steps': 300,
}


class HSKDataset(Dataset):
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


class KGMNN(nn.Module):
    """
    简化 KGMNN：
    - 文本通道：多尺度 CNN + BiLSTM + Attention
    - 概念通道：对文本通道输出做 self-attention 作为概念表示
    - 两通道拼接后分类
    """

    def __init__(self, vocab_size, embed_dim=200, hidden_dim=256,
                 conv_channels=96, num_classes=7, max_length=200,
                 attention_heads=8, num_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(0.2)

        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, conv_channels, k, padding='same')
            for k in (2, 3, 4)
        ])
        self.conv_norms = nn.ModuleList([
            nn.BatchNorm1d(conv_channels) for _ in range(3)
        ])
        self.lstm = nn.LSTM(
            conv_channels * 3, hidden_dim // 2,
            num_layers=num_layers, batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.text_attn = nn.MultiheadAttention(
            hidden_dim, attention_heads, dropout=0.1, batch_first=True
        )
        self.text_norm = nn.LayerNorm(hidden_dim)

        self.concept_attn = nn.MultiheadAttention(
            hidden_dim, attention_heads, dropout=0.1, batch_first=True
        )
        self.concept_norm = nn.LayerNorm(hidden_dim)

        self.gamma = nn.Parameter(torch.tensor(0.5))

        fused_dim = hidden_dim * 4
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.embed_dropout(x)

        x_t = x.transpose(1, 2)
        conv_feats = []
        for conv, norm in zip(self.convs, self.conv_norms):
            c = torch.nn.functional.gelu(norm(conv(x_t)))
            conv_feats.append(c)
        c = torch.cat(conv_feats, dim=1).transpose(1, 2)

        lstm_out, _ = self.lstm(c)
        key_padding_mask = ~attention_mask.bool() if attention_mask is not None else None
        text_attn_out, _ = self.text_attn(
            lstm_out, lstm_out, lstm_out,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        text_repr = self.text_norm(lstm_out + text_attn_out)
        text_pool = text_repr.mean(dim=1)

        concept_attn_out, _ = self.concept_attn(
            text_repr, text_repr, text_repr,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        concept_repr = self.concept_norm(text_repr + concept_attn_out)
        concept_pool = concept_repr.mean(dim=1)

        fused = torch.cat([
            text_pool,
            concept_pool,
            self.gamma * text_pool,
            (1 - self.gamma) * concept_pool,
        ], dim=1)
        logits = self.classifier(fused)
        return logits


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_json_atomic(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)
        f.flush()
        os.fsync(f.fileno())
    with open(tmp, 'r', encoding='utf-8') as f:
        json.load(f)
    os.replace(tmp, path)


def train_one_seed(seed, config=CONFIG):
    print(f"\n{'='*60}")
    print(f"KGMNN Training with seed = {seed}")
    print(f"{'='*60}")

    device = get_device()
    set_seed(seed)

    texts, labels, sources = load_raw_data(BASE_PATH)
    train_t, train_l, val_t, val_l, test_t, test_l = split_by_source(
        texts, labels, sources, seed=seed
    )
    train_t, train_l = augment_training_only(
        train_t, train_l, target=config['augment_target'], seed=seed
    )
    vocab = build_vocab(train_t, config['vocab_size'])

    train_ds = HSKDataset(train_t, train_l, vocab, config['max_length'])
    val_ds = HSKDataset(val_t, val_l, vocab, config['max_length'])
    test_ds = HSKDataset(test_t, test_l, vocab, config['max_length'])

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'],
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = KGMNN(
        vocab_size=len(vocab),
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        conv_channels=config['conv_channels'],
        num_classes=config['num_classes'],
        max_length=config['max_length'],
        attention_heads=config['attention_heads'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
    ).to(device)

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

    counts = Counter(train_l)
    weights = torch.tensor(
        [1.0 / np.sqrt(counts.get(i, 1)) for i in range(7)],
        dtype=torch.float32
    ).to(device)
    weights = weights / weights.sum() * 7
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    best_val_acc = 0.0
    best_state = None
    patience_cnt = 0
    train_start = time.time()

    for epoch in range(config['epochs']):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}",
                    leave=False)
        for batch in pbar:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            y = batch['label'].to(device)

            optimizer.zero_grad()
            logits = model(ids, mask)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(
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
            print(f"  Epoch {epoch+1}: val_acc={val_acc:.4f} (best)")
        else:
            patience_cnt += 1
            print(f"  Epoch {epoch+1}: val_acc={val_acc:.4f} "
                  f"(patience {patience_cnt}/{config['patience']})")
            if patience_cnt >= config['patience']:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    train_time = time.time() - train_start

    model.load_state_dict(best_state)
    model.eval()
    test_preds, test_true, test_probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device)
            )
            probs = torch.softmax(logits, dim=1)
            test_preds.extend(logits.argmax(1).cpu().numpy())
            test_true.extend(batch['label'].numpy())
            test_probs.extend(probs.cpu().numpy())

    metrics = compute_all_metrics(test_true, test_preds)
    metrics['seed'] = int(seed)
    metrics['model'] = 'KGMNN'
    metrics['best_val_acc'] = float(best_val_acc)
    metrics['train_time_sec'] = float(train_time)
    metrics['test_probs'] = [p.tolist() for p in test_probs]
    metrics['true_labels'] = [int(x) for x in test_true]

    print(f"\nSeed {seed} KGMNN results:")
    print(f"  Test accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 weighted: {metrics['f1_weighted']:.4f}")
    print(f"  Train time: {train_time/60:.1f} min")
    return metrics


def main():
    print("=" * 60)
    print("KGMNN Retraining (baseline, seed 123)")
    print(f"Seeds: {SEEDS}")
    print("=" * 60)

    all_results = []
    for seed in SEEDS:
        path = os.path.join(OUTPUT_DIR, f'kgmnn_seed_{seed}.json')
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    d = json.load(f)
                if 'test_probs' in d and 'true_labels' in d:
                    print(f"[Skip] seed {seed} already done.")
                    all_results.append(d)
                    continue
            except Exception:
                pass

        result = train_one_seed(seed)
        all_results.append(result)
        save_json_atomic(result, path)
        print(f"[Saved] {path}")

    if all_results:
        accs = [r['accuracy'] for r in all_results]
        print("\n" + "=" * 60)
        print(f"KGMNN mean acc: {np.mean(accs)*100:.2f} +/- {np.std(accs)*100:.2f}%")
        print("=" * 60)
        summary = {
            'accuracy_mean': float(np.mean(accs)),
            'accuracy_std': float(np.std(accs)),
            'individual_results': all_results,
        }
        save_json_atomic(summary, os.path.join(OUTPUT_DIR, 'kgmnn_results.json'))
        print(f"[Saved] {os.path.join(OUTPUT_DIR, 'kgmnn_results.json')}")


if __name__ == '__main__':
    main()