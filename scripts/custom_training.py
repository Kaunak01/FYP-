"""CUSTOM 3 — Custom Early Stopping class."""

class EarlyStopping:
    """Tracks validation loss and stops training when it hasn't improved for patience epochs.
    No library dependencies."""

    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_epoch = 0
        self.best_state = None
        self.should_stop = False
        self.history = []

    def step(self, val_loss, epoch, model_state=None):
        """Call after each epoch with validation loss.
        Returns True if training should stop."""
        self.history.append(val_loss)

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            if model_state is not None:
                self.best_state = {k: v.clone() for k, v in model_state.items()}
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
            return False

    def status(self, epoch, val_loss):
        """Print epoch status."""
        marker = " *" if val_loss <= self.best_loss else f" (patience {self.counter}/{self.patience})"
        print(f"  Epoch {epoch:3d} — val_loss: {val_loss:.6f}{marker}")


if __name__ == '__main__':
    # Simulate training
    losses = [0.5, 0.45, 0.43, 0.42, 0.425, 0.43, 0.435, 0.44]
    es = EarlyStopping(patience=3)
    for i, loss in enumerate(losses):
        es.status(i+1, loss)
        if es.step(loss, i+1):
            print(f"  --> Early stopping triggered at epoch {i+1}")
            print(f"  --> Best epoch: {es.best_epoch}, best loss: {es.best_loss:.6f}")
            break
