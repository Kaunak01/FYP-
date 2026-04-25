"""CUSTOM 2 — Custom Stratified Sampler. NO sklearn."""
import random

def stratified_split(X, y, test_size=0.2, random_state=42):
    """Split data maintaining class proportions. No sklearn."""
    rng = random.Random(random_state)

    # Separate indices by class
    class_indices = {}
    for i, label in enumerate(y):
        class_indices.setdefault(int(label), []).append(i)

    train_idx, test_idx = [], []
    for label, indices in class_indices.items():
        indices = indices.copy()
        rng.shuffle(indices)
        n_test = int(len(indices) * test_size)
        test_idx.extend(indices[:n_test])
        train_idx.extend(indices[n_test:])

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    X_train = [X[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]

    return X_train, X_test, y_train, y_test, train_idx, test_idx


def stratified_kfold(y, n_splits=5, random_state=42):
    """Generate fold indices maintaining class proportions. No sklearn.
    Yields (train_indices, val_indices) for each fold."""
    rng = random.Random(random_state)

    # Separate indices by class
    class_indices = {}
    for i, label in enumerate(y):
        class_indices.setdefault(int(label), []).append(i)

    # Shuffle within each class
    for label in class_indices:
        rng.shuffle(class_indices[label])

    # Assign each index to a fold
    fold_assignments = [0] * len(y)
    for label, indices in class_indices.items():
        for i, idx in enumerate(indices):
            fold_assignments[idx] = i % n_splits

    # Generate train/val splits
    folds = []
    for fold in range(n_splits):
        val_idx = [i for i, f in enumerate(fold_assignments) if f == fold]
        train_idx = [i for i, f in enumerate(fold_assignments) if f != fold]
        folds.append((train_idx, val_idx))

    return folds


if __name__ == '__main__':
    # Test with small data
    y = [0]*100 + [1]*10
    folds = stratified_kfold(y, n_splits=5)
    for i, (tr, vl) in enumerate(folds):
        fraud_tr = sum(y[j] for j in tr)
        fraud_vl = sum(y[j] for j in vl)
        print(f"Fold {i+1}: train={len(tr)} ({fraud_tr} fraud, {100*fraud_tr/len(tr):.1f}%), val={len(vl)} ({fraud_vl} fraud, {100*fraud_vl/len(vl):.1f}%)")
