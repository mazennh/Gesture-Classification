import torch
from typing import Dict, Optional
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score
import warnings

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"
device

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               metrics: torchmetrics.MetricCollection,
               device: torch.device = 'cpu'):
    """
    Performs a single training epoch step.

    Args:
        model: PyTorch model to train.
        dataloader: DataLoader containing training data.
        loss_fn: Loss function (e.g., nn.BCEWithLogitsLoss).
        optimizer: Optimizer (e.g., SGD, Adam).
        metrics: Collection of metrics (e.g., accuracy, precision, recall, F1)
            to calculate on the predictions..
        device: Target device (e.g., 'cuda', 'cpu'). Defaults to 'cpu'.

    Returns:
        tuple: (average_train_loss, average_train_metrics)
    """

    model.train()
    metrics.to(device)
    metrics.reset()
    train_loss = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_logits = model(X)
        loss = loss_fn(y_logits, y)

        # 2. Predictions
        y_pred = torch.argmax(y_logits,dim=1)

        # 3. Calculate metrics
        metrics.update(y_pred, y)
        train_loss += loss.item()

        # 4. Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Get average loss per batch
    train_loss = train_loss / len(dataloader)
    results = metrics.compute()

    return train_loss, results

def test_step(dataloader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              metrics: torchmetrics.MetricCollection,
              device: torch.device = "cpu"):
    """
    Evaluates the model on unseen data (Validation/Test set).

    Args:
        dataloader: Dataloader for testing data.
        model: PyTorch model.
        loss_fn: Loss function (CrossEntropyLoss).
        metrics: Collection of metrics (accuracy, precision, etc.).
        device: Target device (e.g. 'cuda', 'cpu').

    Returns:
        tuple: (test_loss, results_dict, y_pred_tensor, y_true_tensor)
    """
    model.eval()
    metrics.to(device)
    metrics.reset()
    
    test_loss = 0
    y_preds = []
    y_targets = []
    y_probs = []

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            y_logits = model(X) # Shape: [Batch_Size, Num_Classes]
            loss = loss_fn(y_logits, y)

            # 2. Predictions
            y_prob = torch.softmax(y_logits, dim=1)
            y_pred = torch.argmax(y_prob, dim=1)

            # 3. Update Metrics
            metrics.update(y_prob, y)
            test_loss += loss.item()
            y_preds.append(y_pred.cpu())
            y_targets.append(y.cpu())
            y_probs.append(y_prob.cpu())

    # Calculate average loss
    test_loss = test_loss / len(dataloader)
    
    results = metrics.compute()

    # Concatenate all batches into one long tensor
    y_pred_tensor = torch.cat(y_preds).long()
    y_true_tensor = torch.cat(y_targets).long()
    y_prob_tensor = torch.cat(y_probs).float()

    return test_loss, results, y_pred_tensor, y_true_tensor,y_prob_tensor

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          num_classes: int,
          best_model: str,
          experiment_name: str,
          scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
          device: torch.device = "cpu",
          patience: int = 5,
          epochs: int = 5) -> Dict[str, any]:

    """
    Trains and validate a PyTorch model for Multi-Class Classification with 
    Early Stopping, TensorBoard logging, and LR Scheduling.
    
    Args:
        model: PyTorch model to train.
        train_dataloader: DataLoader for training data.
        val_dataloader: DataLoader for testing/validation data.
        optimizer: Optimizer (e.g. SGD, Adam).
        loss_fn: Loss function (e.g. BCEWithLogitsLoss).
        scheduler: Learning rate scheduler (optional). Defaults to None.
        device: Target device (e.g. 'cuda', 'cpu'). Defaults to 'cpu'.
        epochs: Number of training epochs. Defaults to 5.

    Returns:
        dict: Dictionary containing training and validation metrics.

    Example:
      history=train(model=model,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    num_classes = 9,
                    scheduler=scheduler,
                    device=device,
                    patience=10,
                    experiment_name = "My_Experiment",
                    epochs=100)
    """

    best_val_loss = float('inf')
    counter = 0
    best_model_path = best_model
    
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    writer = SummaryWriter(log_dir=f"runs/{experiment_name}")

    metric_collection = MetricCollection({
          'acc': Accuracy(task="multiclass", num_classes=num_classes),
          'prec': Precision(task="multiclass", num_classes=num_classes, average='macro'),
          'rec': Recall(task="multiclass", num_classes=num_classes, average='macro'),
          'f1': F1Score(task="multiclass", num_classes=num_classes, average='macro')
      })

    train_metrics = metric_collection.clone().to(device)
    val_metrics = metric_collection.clone().to(device)

    model.to(device)

    try:
        dummy_input, _ = next(iter(train_dataloader))
        writer.add_graph(model, dummy_input.to(device))
    except Exception as e:
        print(f"TensorBoard Graph skipped: {e}")

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        
        train_loss, train_res = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           metrics=train_metrics,
                                           device=device)

        val_loss, val_res, _, _, _ = test_step(model=model,
                                            dataloader=val_dataloader,
                                            loss_fn=loss_fn,
                                            metrics=val_metrics,
                                            device=device)

        # Extract values for logging
        train_acc = train_res['acc'].item() * 100
        train_f1  = train_res['f1'].item() * 100
        train_rec = train_res['rec'].item() * 100
        train_prec = train_res['prec'].item() * 100
        
        val_acc = val_res['acc'].item() * 100
        val_prec = val_res['prec'].item() * 100
        val_rec = val_res['rec'].item() * 100
        val_f1  = val_res['f1'].item() * 100

        # --- TensorBoard Logging ---
        writer.add_scalars('Loss', {'Train': train_loss, 'Val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'Train': train_acc, 'Val': val_acc}, epoch)
        writer.add_scalars('F1_Score', {'Train': train_f1, 'Val': val_f1}, epoch)
        writer.add_scalar('Recall/Val', val_rec, epoch)
        writer.add_scalar('Precision/VAL', val_prec, epoch)

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        print(
            f"Epoch: {epoch+1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% | "
            f"Val F1: {val_f1:.2f}%"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # 4. Learning Rate Scheduler
        if scheduler is not None:
            # If using ReduceLROnPlateau, we must pass the validation loss
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # 5. Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("Best Model Saved...")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    writer.close()
    return history
