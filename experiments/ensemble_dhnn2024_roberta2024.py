"""
严格同划分集成：DHNN seed 2024 + RoBERTa seed 2024
- 只用 split_by_source(seed=2024) 下的两个模型
- 逐样本概率平均
- 三重校验：长度一致、true_labels 完全一致、各自 acc 与 json 一致
- 输出 ensemble_dhnn2024_roberta2024.json
"""

import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
SEED = 2024


def load_json(path):
    if not os.path.exists(path):
        print(f"[Error] Not found: {path}")
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[Error] Invalid JSON {path}: {e}")
        return None


def check_model(d, name, seed):
    if d is None:
        return False
    for k in ('test_probs', 'true_labels', 'accuracy'):
        if k not in d:
            print(f"[Error] {name} 缺少字段: {k}")
            return False
    if int(d.get('seed', -1)) != int(seed):
        print(f"[Error] {name} seed 不匹配: json={d.get('seed')}, expected={seed}")
        return False
    n_probs = len(d['test_probs'])
    n_true = len(d['true_labels'])
    if n_probs != n_true or n_probs == 0:
        print(f"[Error] {name} 长度异常: probs={n_probs}, labels={n_true}")
        return False
    dims = {len(row) for row in d['test_probs'][:10]}
    if len(dims) != 1:
        print(f"[Error] {name} 概率维度不一致: {dims}")
        return False
    print(f"[OK] {name}: seed={seed}, n={n_probs}, dim={dims.pop()}, "
          f"acc={d['accuracy']:.4f}")
    return True


def main():
    seed = SEED

    # ---------- 1. DHNN seed 2024 ----------
    dhnn_path = os.path.join(RESULTS_DIR, f'main_results_seed_{seed}.json')
    dhnn = load_json(dhnn_path)
    if not check_model(dhnn, f'DHNN seed {seed}', seed):
        return

    # ---------- 2. RoBERTa seed 2024 ----------
    rb_path = os.path.join(RESULTS_DIR, f'roberta_seed_{seed}.json')
    rb = load_json(rb_path)
    if not check_model(rb, f'RoBERTa seed {seed}', seed):
        return

    # ---------- 3. 长度与标签一致性 ----------
    n_dhnn = len(dhnn['test_probs'])
    n_rb = len(rb['test_probs'])
    if n_dhnn != n_rb:
        print(f"[Error] 样本数不同: DHNN={n_dhnn}, RoBERTa={n_rb}")
        print("  说明两者不是同一 split_by_source 划分，不能集成。")
        return

    y_dhnn = np.array(dhnn['true_labels'], dtype=np.int64)
    y_rb = np.array(rb['true_labels'], dtype=np.int64)
    if not np.array_equal(y_dhnn, y_rb):
        diff = int((y_dhnn != y_rb).sum())
        print(f"[Error] true_labels 不一致，差异 {diff} 条。")
        print("  可能原因：两者不是同一 split_by_source(seed=2024) 划分。")
        return
    print(f"[OK] true_labels 完全一致（{len(y_dhnn)} 条）")
    true_labels = y_dhnn

    # ---------- 4. 概率与自身 acc 校验 ----------
    p_dhnn = np.array(dhnn['test_probs'], dtype=np.float64)
    p_rb = np.array(rb['test_probs'], dtype=np.float64)

    if p_dhnn.shape != p_rb.shape:
        print(f"[Error] 概率形状不同: DHNN={p_dhnn.shape}, RoBERTa={p_rb.shape}")
        return

    if not np.allclose(p_dhnn.sum(axis=1), 1.0, atol=1e-3):
        print("[Warning] DHNN 概率行和偏离 1")
    if not np.allclose(p_rb.sum(axis=1), 1.0, atol=1e-3):
        print("[Warning] RoBERTa 概率行和偏离 1")

    acc_dhnn_check = accuracy_score(true_labels, p_dhnn.argmax(axis=1))
    if abs(acc_dhnn_check - dhnn['accuracy']) > 1e-4:
        print(f"[Error] DHNN 自身 acc 不一致: json={dhnn['accuracy']:.6f}, "
              f"recomputed={acc_dhnn_check:.6f}")
        return
    print(f"[OK] DHNN 自身 acc 校验通过: {acc_dhnn_check:.4f}")

    acc_rb_check = accuracy_score(true_labels, p_rb.argmax(axis=1))
    if abs(acc_rb_check - rb['accuracy']) > 1e-4:
        print(f"[Error] RoBERTa 自身 acc 不一致: json={rb['accuracy']:.6f}, "
              f"recomputed={acc_rb_check:.6f}")
        return
    print(f"[OK] RoBERTa 自身 acc 校验通过: {acc_rb_check:.4f}")

    # ---------- 5. 等权概率平均 ----------
    print("\n" + "=" * 60)
    print(f"Ensemble: DHNN seed {seed} + RoBERTa seed {seed}")
    print("=" * 60)
    print(f"  DHNN    acc={acc_dhnn_check*100:.2f}%  "
          f"F1w={f1_score(true_labels, p_dhnn.argmax(1), average='weighted')*100:.2f}%")
    print(f"  RoBERTa acc={acc_rb_check*100:.2f}%  "
          f"F1w={f1_score(true_labels, p_rb.argmax(1), average='weighted')*100:.2f}%")

    avg_probs = (p_dhnn + p_rb) / 2.0
    preds = avg_probs.argmax(axis=1)

    acc = accuracy_score(true_labels, preds)
    f1w = f1_score(true_labels, preds, average='weighted')
    f1m = f1_score(true_labels, preds, average='macro')
    cm = confusion_matrix(true_labels, preds, labels=list(range(7)))

    print(f"\n  Ensemble acc={acc*100:.2f}%  "
          f"F1w={f1w*100:.2f}%  F1m={f1m*100:.2f}%")
    print("  Confusion matrix:")
    print(cm)

    result = {
        'composition': {
            'DHNN_seed': seed,
            'RoBERTa_seed': seed,
            'n_models': 2,
            'fusion': 'probability_average',
        },
        'n_samples': int(len(true_labels)),
        'DHNN_accuracy': float(acc_dhnn_check),
        'DHNN_f1_weighted': float(f1_score(true_labels, p_dhnn.argmax(1),
                                           average='weighted')),
        'RoBERTa_accuracy': float(acc_rb_check),
        'RoBERTa_f1_weighted': float(f1_score(true_labels, p_rb.argmax(1),
                                              average='weighted')),
        'ensemble_accuracy': float(acc),
        'ensemble_f1_weighted': float(f1w),
        'ensemble_f1_macro': float(f1m),
        'confusion_matrix': cm.tolist(),
    }

    out_path = os.path.join(RESULTS_DIR,
                            f'ensemble_dhnn{seed}_roberta{seed}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] {out_path}")


if __name__ == '__main__':
    main()