"""
RoBERTa-wwm-ext 微调（仅跑 seed 42，patience=5）
- SEEDS = [42]
- epochs=20, patience=5
- batch_size=8，8 个 CPU 线程，实时进度条
- 支持 epoch 级断点续跑
- true_labels 转 Python int，JSON 原子写入 + 回读校验
- 不删除检查点
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
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup
)

torch.set_num_threads(8)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.seeding import set_seed, get_device
from utils.data_loader import (
    load_raw_data, split_by_source, augment_training_only
)
from utils.metrics import compute_all_metrics


BASE_PATH = r"C:\Users\DF-Lenovo\Desktop\数据集"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 本次只跑 seed 42
SEEDS = [42]

CONFIG = {
    'model_name': 'hfl/chinese-roberta-wwm-ext',
    'max_length': 200,
    'batch_size': 8,
    'epochs': 20,            # 保险上限
    'patience': 5,           # 连续 5 轮不提升就停
    'lr': 2e-5,
    'weight_decay': 1e-4,
    'warmup_steps': 100,
    'num_classes': 7,
    'augment_target': 800,
}


class TextDataset(Dataset):
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
    print(f"RoBERTa Training with seed = {seed}")
    print(f"CPU threads: {torch.get_num_threads()}")
    print(f"{'='*60}")

    device = get_device()
    set_seed(seed)

    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'roberta_seed_{seed}_latest.pt')

    texts, labels, sources = load_raw_data(BASE_PATH)
    train_t, train_l, val_t, val_l, test_t, test_l = split_by_source(
        texts, labels, sources, seed=seed
    )
    train_t, train_l = augment_training_only(
        train_t, train_l, target=config['augment_target'], seed=seed
    )

    print(f"  Loading {config['model_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model_name'], num_labels=config['num_classes']
    ).to(device)

    train_ds = TextDataset(train_t, train_l, tokenizer, config['max_length'])
    val_ds = TextDataset(val_t, val_l, tokenizer, config['max_length'])
    test_ds = TextDataset(test_t, test_l, tokenizer, config['max_length'])

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    print(f"  Train batches per epoch: {len(train_loader)}")
    print(f"  Train samples: {len(train_ds)}")

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

    start_epoch = 0
    best_val_acc = 0.0
    best_state = None
    patience_cnt = 0

    if os.path.exists(checkpoint_path):
        print(f"  [Resume] Found checkpoint")
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_val_acc = ckpt['best_val_acc']
            patience_cnt = ckpt['patience_cnt']
            best_state = ckpt.get('best_state', None)
            print(f"  Resuming from epoch {start_epoch}, best={best_val_acc:.4f}")
        except Exception as e:
            print(f"  [Warning] Checkpoint load failed: {e}. Starting fresh.")

    train_start = time.time()
    epoch_times = []

    for epoch in range(start_epoch, config['epochs']):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            train_loader,
            desc=f"Seed {seed} | Epoch {epoch+1}/{config['epochs']}",
            leave=False,
            ncols=100
        )

        for batch in pbar:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels_batch = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=ids, attention_mask=mask, labels=labels_batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.logits, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()

            avg_loss = total_loss / (pbar.n + 1)
            train_acc = correct / total
            current_lr = optimizer.param_groups[0]['lr']

            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{train_acc:.3f}',
                'lr': f'{current_lr:.2e}'
            })

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

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
            print(f"  ✅ Epoch {epoch+1}: val_acc={val_acc:.4f} (best) | "
                  f"time={epoch_time/60:.1f}min")
        else:
            patience_cnt += 1
            print(f"  ⏳ Epoch {epoch+1}: val_acc={val_acc:.4f} | "
                  f"time={epoch_time/60:.1f}min | "
                  f"patience={patience_cnt}/{config['patience']}")

        if (epoch + 1) % 2 == 0 or epoch == config['epochs'] - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'patience_cnt': patience_cnt,
                'best_state': best_state,
            }, checkpoint_path)
            print(f"  💾 Checkpoint saved at epoch {epoch+1}")

        if patience_cnt >= config['patience']:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    train_time = time.time() - train_start

    print(f"\n  Training time: {train_time/60:.1f} min")
    if epoch_times:
        print(f"  Avg epoch time: {np.mean(epoch_times)/60:.1f} min")

    if best_state is None:
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    test_preds, test_true = [], []
    test_probs = []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device)
            )
            probs = torch.softmax(outputs.logits, dim=1)
            test_preds.extend(outputs.logits.argmax(1).cpu().numpy())
            test_true.extend(batch['label'].numpy())
            test_probs.extend(probs.cpu().numpy())

    metrics = compute_all_metrics(test_true, test_preds)
    metrics['seed'] = int(seed)
    metrics['model'] = 'RoBERTa'
    metrics['best_val_acc'] = float(best_val_acc)
    metrics['train_time_sec'] = float(train_time)
    metrics['test_probs'] = [p.tolist() for p in test_probs]
    # ★ 关键：numpy.int64 -> Python int
    metrics['true_labels'] = [int(x) for x in test_true]

    print(f"\nSeed {seed} RoBERTa results:")
    print(f"  Test accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 weighted: {metrics['f1_weighted']:.4f}")
    print(f"  Train time: {train_time/60:.1f} min")

    return metrics


def main():
    print("="*60)
    print("RoBERTa-wwm-ext Fine-tuning (seed 42, patience=5)")
    print(f"Model: {CONFIG['model_name']}")
    print(f"Seeds: {SEEDS}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Epochs: {CONFIG['epochs']}, patience: {CONFIG['patience']}")
    print(f"CPU threads: {torch.get_num_threads()}")
    print(f"Device: CPU")
    print("="*60)

    all_results = []

    for seed in SEEDS:
        result_path = os.path.join(OUTPUT_DIR, f'roberta_seed_{seed}.json')

        existing = load_json_if_valid(result_path)
        if existing is not None:
            print(f"\n[Skip] RoBERTa seed {seed} already done (valid JSON).")
            all_results.append(existing)
            continue
        elif os.path.exists(result_path):
            print(f"\n[Repair] RoBERTa seed {seed} JSON is broken, will re-run.")

        try:
            result = train_one_seed(seed)
            all_results.append(result)

            save_json_atomic(result, result_path)
            print(f"[Saved] {result_path}")

            check = load_json_if_valid(result_path)
            if check is None:
                raise RuntimeError(f"Saved JSON is still invalid: {result_path}")
            print(f"[Verified] {result_path} | "
                  f"true_labels={len(check['true_labels'])}, "
                  f"probs={len(check['test_probs'])}, "
                  f"acc={check['accuracy']:.4f}")

        except Exception as e:
            print(f"RoBERTa seed {seed} failed: {e}")
            import traceback
            traceback.print_exc()

    if all_results:
        accs = [r['accuracy'] for r in all_results]
        print(f"\n{'='*60}")
        print(f"RoBERTa summary (this run):")
        print(f"  Accuracy: {np.mean(accs)*100:.2f} +/- {np.std(accs)*100:.2f}%")
        print(f"{'='*60}")

        summary = {
            'accuracy_mean': float(np.mean(accs)),
            'accuracy_std': float(np.std(accs)),
            'individual_results': all_results,
        }
        summary_path = os.path.join(OUTPUT_DIR, 'roberta_results.json')
        save_json_atomic(summary, summary_path)
        print(f"[Saved] {summary_path}")

        check = load_json_if_valid(summary_path)
        if check is None:
            raise RuntimeError(f"Saved summary JSON is invalid: {summary_path}")
        print(f"[Verified] {summary_path} | "
              f"n_models={len(check['individual_results'])}, "
              f"acc_mean={check['accuracy_mean']:.4f}")


if __name__ == '__main__':
    main()