"""CUSTOM 1 — Metrics Module. ALL from scratch, NO sklearn, NO numpy shortcuts."""
import math

def calculate_confusion_matrix(y_true, y_pred):
    """Return dict with TP, FP, FN, TN."""
    tp = fp = fn = tn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1: tp += 1
        elif yt == 0 and yp == 1: fp += 1
        elif yt == 1 and yp == 0: fn += 1
        else: tn += 1
    return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn}

def calculate_precision(y_true, y_pred):
    cm = calculate_confusion_matrix(y_true, y_pred)
    denom = cm['TP'] + cm['FP']
    return cm['TP'] / denom if denom > 0 else 0.0

def calculate_recall(y_true, y_pred):
    cm = calculate_confusion_matrix(y_true, y_pred)
    denom = cm['TP'] + cm['FN']
    return cm['TP'] / denom if denom > 0 else 0.0

def calculate_f1(y_true, y_pred):
    p = calculate_precision(y_true, y_pred)
    r = calculate_recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def calculate_specificity(y_true, y_pred):
    cm = calculate_confusion_matrix(y_true, y_pred)
    denom = cm['TN'] + cm['FP']
    return cm['TN'] / denom if denom > 0 else 0.0

def calculate_mcc(y_true, y_pred):
    """Matthews Correlation Coefficient."""
    cm = calculate_confusion_matrix(y_true, y_pred)
    tp, fp, fn, tn = cm['TP'], cm['FP'], cm['FN'], cm['TN']
    num = tp * tn - fp * fn
    denom = math.sqrt((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn))
    return num / denom if denom > 0 else 0.0

def calculate_roc_points(y_true, y_prob):
    """Return list of (fpr, tpr) pairs at different thresholds."""
    # Get sorted unique thresholds
    thresholds = sorted(set(y_prob), reverse=True)
    # Add boundary thresholds
    points = []
    for thresh in [max(y_prob) + 0.001] + thresholds + [min(y_prob) - 0.001]:
        tp = fp = fn = tn = 0
        for yt, yp in zip(y_true, y_prob):
            pred = 1 if yp >= thresh else 0
            if yt == 1 and pred == 1: tp += 1
            elif yt == 0 and pred == 1: fp += 1
            elif yt == 1 and pred == 0: fn += 1
            else: tn += 1
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        points.append((fpr, tpr))
    return points

def calculate_auc(points):
    """AUC using trapezoidal rule. Points are (x, y) pairs."""
    # Sort by x
    points = sorted(points, key=lambda p: p[0])
    auc_val = 0.0
    for i in range(1, len(points)):
        x0, y0 = points[i-1]
        x1, y1 = points[i]
        auc_val += (x1 - x0) * (y0 + y1) / 2.0
    return auc_val

def calculate_pr_curve(y_true, y_prob):
    """Return list of (recall, precision) pairs at different thresholds."""
    thresholds = sorted(set(y_prob), reverse=True)
    points = []
    for thresh in thresholds:
        tp = fp = fn = 0
        for yt, yp in zip(y_true, y_prob):
            pred = 1 if yp >= thresh else 0
            if yt == 1 and pred == 1: tp += 1
            elif yt == 0 and pred == 1: fp += 1
            elif yt == 1 and pred == 0: fn += 1
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        points.append((rec, prec))
    return points

def calculate_average_precision(y_true, y_prob):
    """Average precision score."""
    pr_points = calculate_pr_curve(y_true, y_prob)
    pr_points = sorted(pr_points, key=lambda p: p[0])
    ap = 0.0
    prev_recall = 0.0
    for recall, precision in pr_points:
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return ap


if __name__ == '__main__':
    # Quick self-test with small data
    y_t = [0,0,0,0,1,1,1,1,1,0]
    y_p = [0,0,1,0,1,1,0,1,1,1]
    cm = calculate_confusion_matrix(y_t, y_p)
    print(f"CM: {cm}")
    print(f"Precision: {calculate_precision(y_t, y_p):.4f}")
    print(f"Recall: {calculate_recall(y_t, y_p):.4f}")
    print(f"F1: {calculate_f1(y_t, y_p):.4f}")
    print(f"Specificity: {calculate_specificity(y_t, y_p):.4f}")
    print(f"MCC: {calculate_mcc(y_t, y_p):.4f}")
