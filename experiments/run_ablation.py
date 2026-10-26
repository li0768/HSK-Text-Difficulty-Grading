"""
DHNN 消融实验（与主实验同配置）
- 6 个变体：Complete, -CNN, -BiLSTM, -Attention, -Auxiliary, -FocalLoss
- 3 个 seed：42, 123, 2024
- 保存 test_probs / true_labels，JSON 原子写入 + 回读校验
- 支持断点续跑：已完成的 variant+seed 会跳过
- 汇总到 ablation_results.json
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
from models.dhnn import DHNN, FocalLoss, CrossEntropyLossWrapper


BASE_PATH = r"C:\Users\DF-Lenovo\Desktop\数据集"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEEDS = [42, 123, 2024]

# ★ 必须和主实验（run_main_with_probs.py）完全一致
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

VARIANTS = {
    'Complete':   {'use_cnn': True,  'use_lstm': True,  'use_attention': True,  'use_aux': True,  'loss': 'focal'},
    '-CNN':       {'use_cnn': False, 'use_lstm': True,  'use_attention': True,  'use_aux': True,  'loss': 'focal'},
    '-BiLSTM':    {'use_cnn': True,  'use_lstm': False, 'use_attention': True,  'use_aux': True,  'loss': 'focal'},
    '-Attention': {'use_cnn': True,  'use_lstm': True,  'use_attention': False, 'use_aux': True,  'loss': 'focal'},
    '-Auxiliary': {'use_cnn': True,  'use_lstm': True,  'use_attention': True,  'use_aux': False, 'loss': 'focal'},
    '-FocalLoss': {'use_cnn': True,  'use_lstm': True,  'use_attention': True,  'use_aux': True,  'loss': 'ce'},
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


def train_one_variant_seed(variant_name, seed, vcfg, config=CONFIG):
    print(f"\n{'='*60}")
    print(f"Ablation {variant_name} | seed = {seed}")
    print(f"  cfg: {vcfg}")
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
        dropout=config['dropout'],
        use_cnn=vcfg['use_cnn'],
        use_lstm=vcfg['use_lstm'],
        use_attention=vcfg['use_attention'],
        use_aux=vcfg['use_aux'],
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

    if vcfg['loss'] == 'focal':
        criterion = FocalLoss(
            alpha=weights,
            gamma=config['focal_gamma'],
            smoothing=config['label_smoothing'],
            use_smoothing=True
        )
    else:
        criterion = CrossEntropyLossWrapper(
            alpha=weights,
            smoothing=config['label_smoothing']
        )

    best_val_acc = 0.0
    best_state = None
    patience_cnt = 0
    train_start = time.time()

    for epoch in range(config['epochs']):
        model.train()
        pbar = tqdm(train_loader, desc=f"{variant_name} s{seed} "
                                       f"E{epoch+1}/{config['epochs']}",
                    leave=False)
        for batch in pbar:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            y = batch['label'].to(device)

            optimizer.zero_grad()
            main_logits, aux_logits = model(ids, mask)
            loss = criterion(main_logits, y)
            if vcfg['use_aux'] and aux_logits is not None:
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
    metrics['variant'] = variant_name
    metrics['seed'] = int(seed)
    metrics['best_val_acc'] = float(best_val_acc)
    metrics['train_time_sec'] = float(train_time)
    metrics['n_params'] = int(sum(p.numel() for p in model.parameters()
                                  if p.requires_grad))
    metrics['test_probs'] = [p.tolist() for p in test_probs]
    metrics['true_labels'] = [int(x) for x in test_true]
    metrics['variant_config'] = vcfg

    print(f"\n{variant_name} seed {seed}: "
          f"acc={metrics['accuracy']:.4f}  "
          f"f1w={metrics['f1_weighted']:.4f}  "
          f"f1m={metrics['f1_macro']:.4f}")
    return metrics


def summarize_variant(all_results):
    accs = [r['accuracy'] for r in all_results]
    f1ws = [r['f1_weighted'] for r in all_results]
    f1ms = [r['f1_macro'] for r in all_results]
    return {
        'n_seeds': len(all_results),
        'accuracy_mean': float(np.mean(accs)),
        'accuracy_std': float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        'f1_weighted_mean': float(np.mean(f1ws)),
        'f1_weighted_std': float(np.std(f1ws, ddof=1)) if len(f1ws) > 1 else 0.0,
        'f1_macro_mean': float(np.mean(f1ms)),
        'f1_macro_std': float(np.std(f1ms, ddof=1)) if len(f1ms) > 1 else 0.0,
        'seeds': [r['seed'] for r in all_results],
        'individual_results': all_results,
    }


def main():
    print("=" * 60)
    print("DHNN Ablation Study (same config as main experiment)")
    print(f"Variants: {list(VARIANTS.keys())}")
    print(f"Seeds: {SEEDS}")
    print("=" * 60)

    summary = {}

    for variant_name, vcfg in VARIANTS.items():
        # Complete 变体直接读主实验结果，不重跑
        if variant_name == 'Complete':
            print(f"\n[Load] {variant_name} from main_results_seed_*.json")
            loaded = []
            for seed in SEEDS:
                p = os.path.join(OUTPUT_DIR, f'main_results_seed_{seed}.json')
                d = load_json_if_valid(p)
                if d is None or 'test_probs' not in d:
                    print(f"  [Error] main_results_seed_{seed}.json 缺失或无 probs")
                    break
                loaded.append(d)
            if len(loaded) == len(SEEDS):
                summary[variant_name] = summarize_variant(loaded)
                print(f"  [OK] {variant_name}: "
                      f"acc={summary[variant_name]['accuracy_mean']*100:.2f} "
                      f"± {summary[variant_name]['accuracy_std']*100:.2f}")
            else:
                print(f"  [Skip] {variant_name} 缺少主实验结果")
            continue

        results = []
        for seed in SEEDS:
            path = os.path.join(
                OUTPUT_DIR, f'ablation_{variant_name}_seed_{seed}.json'
            )
            existing = load_json_if_valid(path)
            if existing is not None and 'test_probs' in existing:
                print(f"\n[Skip] {variant_name} seed {seed} already done.")
                results.append(existing)
                continue

            try:
                r = train_one_variant_seed(variant_name, seed, vcfg)
                results.append(r)
                save_json_atomic(r, path)
                print(f"[Saved] {path}")
            except Exception as e:
                print(f"[Error] {variant_name} seed {seed} failed: {e}")
                import traceback
                traceback.print_exc()

        if results:
            summary[variant_name] = summarize_variant(results)
            print(f"\n[Summary] {variant_name}: "
                  f"acc={summary[variant_name]['accuracy_mean']*100:.2f} "
                  f"± {summary[variant_name]['accuracy_std']*100:.2f}")

    out_path = os.path.join(OUTPUT_DIR, 'ablation_results.json')
    save_json_atomic(summary, out_path)
    print(f"\n[Saved] {out_path}")

    print("\n" + "=" * 70)
    print(f"{'Variant':<15}{'Acc (%)':>18}{'F1w (%)':>18}{'F1m (%)':>18}")
    print("-" * 70)
    for name, s in summary.items():
        print(f"{name:<15}"
              f"{s['accuracy_mean']*100:>8.2f} ± {s['accuracy_std']*100:<6.2f}"
              f"{s['f1_weighted_mean']*100:>8.2f} ± {s['f1_weighted_std']*100:<6.2f}"
              f"{s['f1_macro_mean']*100:>8.2f} ± {s['f1_macro_std']*100:<6.2f}")
    print("=" * 70)


if __name__ == '__main__':
    main()