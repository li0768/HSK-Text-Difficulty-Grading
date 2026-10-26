"""
统一指标计算
确保所有指标从原始预测中计算，避免数学不一致
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix
)


def compute_all_metrics(y_true, y_pred, num_classes=7):
    """
    计算所有评估指标
    保证：
    - 混淆矩阵总和 = 测试集大小
    - F1 ≤ max(Precision, Recall)
    - 从混淆矩阵计算的准确率 = 直接计算的准确率
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    acc = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    
    _verify_consistency(y_true, y_pred, acc, cm, p, r, f)
    
    return {
        'accuracy': float(acc),
        'f1_weighted': float(f1_weighted),
        'f1_macro': float(f1_macro),
        'f1_micro': float(f1_micro),
        'per_class_precision': p.tolist(),
        'per_class_recall': r.tolist(),
        'per_class_f1': f.tolist(),
        'per_class_support': support.tolist(),
        'confusion_matrix': cm.tolist(),
        'test_size': len(y_true),
    }


def _verify_consistency(y_true, y_pred, acc, cm, p, r, f):
    """内部一致性验证"""
    n = len(y_true)
    
    assert cm.sum() == n, f"CM sum {cm.sum()} != test size {n}"
    
    acc_from_cm = np.trace(cm) / n
    assert abs(acc_from_cm - acc) < 1e-6, \
        f"Acc mismatch: {acc_from_cm} vs {acc}"
    
    for i, (pi, ri, fi) in enumerate(zip(p, r, f)):
        if not (fi <= max(pi, ri) + 1e-6):
            raise ValueError(
                f"Class {i}: F1={fi:.4f} > max(P={pi:.4f}, R={ri:.4f})"
            )
    
    for i, (pi, ri, fi) in enumerate(zip(p, r, f)):
        if pi + ri > 0:
            fi_expected = 2 * pi * ri / (pi + ri)
            assert abs(fi - fi_expected) < 1e-4, \
                f"Class {i}: F1 formula mismatch {fi} vs {fi_expected}"


def aggregate_over_seeds(results_list):
    """对多个种子的结果进行汇总"""
    accs = [r['accuracy'] for r in results_list]
    f1ws = [r['f1_weighted'] for r in results_list]
    f1ms = [r['f1_macro'] for r in results_list]
    
    p_arr = np.array([r['per_class_precision'] for r in results_list])
    r_arr = np.array([r['per_class_recall'] for r in results_list])
    f_arr = np.array([r['per_class_f1'] for r in results_list])
    cm_arr = np.array([r['confusion_matrix'] for r in results_list])
    
    summary = {
        'n_seeds': len(results_list),
        'accuracy_mean': float(np.mean(accs)),
        'accuracy_std': float(np.std(accs)),
        'accuracy_all': accs,
        'f1_weighted_mean': float(np.mean(f1ws)),
        'f1_weighted_std': float(np.std(f1ws)),
        'f1_macro_mean': float(np.mean(f1ms)),
        'f1_macro_std': float(np.std(f1ms)),
        'per_class_precision_mean': p_arr.mean(axis=0).tolist(),
        'per_class_precision_std': p_arr.std(axis=0).tolist(),
        'per_class_recall_mean': r_arr.mean(axis=0).tolist(),
        'per_class_recall_std': r_arr.std(axis=0).tolist(),
        'per_class_f1_mean': f_arr.mean(axis=0).tolist(),
        'per_class_f1_std': f_arr.std(axis=0).tolist(),
        'confusion_matrix_mean': cm_arr.mean(axis=0).tolist(),
        'individual_results': results_list,
    }
    return summary