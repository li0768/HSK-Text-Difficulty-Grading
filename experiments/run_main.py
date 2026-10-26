"""
DHNN 主实验（带逐样本概率版，用于集成）
- 超参在 CONFIG 里统一改
- 保存 test_probs / true_labels
- JSON 原子写入 + 回读校验
"""

import os
import sys
import json
import time
import numpy as np
import torch
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
from models.dhnn import DHNN, FocalLoss


BASE_PATH = r"C:\Users\DF-Lenovo\Desktop\数据集"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEEDS = [42, 123, 2024]

# ★ 旧 12.7M 配置（按你旧 json 反推，确认后改这里）
CONFIG = {
    'vocab_size': 60000,
    'embed_dim': 300,
    'hidden_dim': 384,
    'num_layers': 3,
    'num_classes': 7,
    'conv_channels': 128,
    'attention_heads': 8,
    'max_length': 200,
    'dropout': 0.4,
    'batch_size': 16,
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'epochs': 80,
    'patience': 20,
    'augment_target': 800,
    'focal_gamma': 2.0,
    'label_smoothing': 0.1,
    'aux_weight': 0.2,
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


def load_json_if_valid(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"  [Warning] Invalid JSON {path}: {e}")
        return None


def train_one_seed(seed, config=CONFIG):
    print(f"\n{'='*60}")
    print(f"Training with seed = {seed}")
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

    model = DHNN(
        vocab_size=len(vocab),
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes'],
        conv_channels=config['conv_channels'],
        attention_heads=config['attention_heads'],
        max_length=config['max_length'],
    ).to(device)

    print(f"  [Model] n_params = "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

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

    criterion = FocalLoss(
        alpha=weights,
        gamma=config['focal_gamma'],
        smoothing=config['label_smoothing'],
        use_smoothing=True
    )

    best_val_acc = 0
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
            main_logits, aux_logits = model(ids, mask)
            loss = criterion(main_logits, y)
            if aux_logits is not None:
                loss = loss + config['aux_weight'] * criterion(aux_logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                main_logits, _ = model(
                    batch['input_ids'].to(device),
                    batch['attention_mask'].to(device)
                )
                val_preds.extend(main_logits.argmax(1).cpu().numpy())
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
            main_logits, _ = model(
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device)
            )
            probs = torch.softmax(main_logits, dim=1)
            test_preds.extend(main_logits.argmax(1).cpu().numpy())
            test_true.extend(batch['label'].numpy())
            test_probs.extend(probs.cpu().numpy())

    metrics = compute_all_metrics(test_true, test_preds)
    metrics['seed'] = int(seed)
    metrics['best_val_acc'] = float(best_val_acc)
    metrics['train_time_sec'] = float(train_time)
    metrics['n_params'] = int(sum(p.numel() for p in model.parameters()
                                  if p.requires_grad))
    metrics['test_probs'] = [p.tolist() for p in test_probs]
    metrics['true_labels'] = [int(x) for x in test_true]

    print(f"\nSeed {seed} results:")
    print(f"  Test accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 weighted: {metrics['f1_weighted']:.4f}")
    print(f"  Test size: {metrics['test_size']}")
    print(f"  Train time: {train_time/60:.1f} min")

    return metrics


def main():
    print("=" * 60)
    print("DHNN Main Experiment (with probs, old 12.7M config)")
    print(f"hidden_dim={CONFIG['hidden_dim']}, "
          f"conv_channels={CONFIG['conv_channels']}, "
          f"num_layers={CONFIG['num_layers']}, "
          f"dropout={CONFIG['dropout']}")
    print(f"lr={CONFIG['lr']}, epochs={CONFIG['epochs']}, "
          f"patience={CONFIG['patience']}")
    print(f"Seeds: {SEEDS}")
    print("=" * 60)

    all_results = []

    for seed in SEEDS:
        path = os.path.join(OUTPUT_DIR, f'main_results_seed_{seed}.json')

        existing = load_json_if_valid(path)
        if existing is not None and 'test_probs' in existing and 'true_labels' in existing:
            print(f"\n[Skip] Seed {seed} already has probs.")
            all_results.append(existing)
            continue
        elif os.path.exists(path):
            print(f"\n[Re-run] Seed {seed}: existing json lacks probs, will retrain.")

        try:
            result = train_one_seed(seed)
            all_results.append(result)
            save_json_atomic(result, path)
            print(f"[Saved] {path}")
        except Exception as e:
            print(f"Seed {seed} failed: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        print("All seeds failed!")
        return

    accs = [r['accuracy'] for r in all_results]
    summary = {
        'accuracy_mean': float(np.mean(accs)),
        'accuracy_std': float(np.std(accs)),
        'completed_seeds': [int(r['seed']) for r in all_results],
        'config': CONFIG,
        'individual_results': all_results,
    }
    save_json_atomic(summary, os.path.join(OUTPUT_DIR, 'main_results_with_probs.json'))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Completed seeds: {summary['completed_seeds']}")
    print(f"Accuracy (mean): {summary['accuracy_mean']*100:.2f} +/- "
          f"{summary['accuracy_std']*100:.2f}%")
    print(f"\nSaved to: {os.path.join(OUTPUT_DIR, 'main_results_with_probs.json')}")


if __name__ == '__main__':
    main()