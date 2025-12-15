import torch
import matplotlib.pyplot as plt
from PIL import Image
import os
from typing import List
import random
import math
from train_utils import test_step
from sklearn.metrics import classification_report
import seaborn as sns
from torchmetrics import (MetricCollection,
                           Accuracy, Precision,
                            Recall, F1Score,
                            AUROC,ROC,ConfusionMatrix)
import warnings

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"
device

def visualize_random_samples(root_path: str,
                             n_samples: int = 8,
                             cols: int = 4):
    """
    Displays n random images from the dataset in a grid layout.
    
    Args:
        root_path (str): The directory containing class subfolders.
        n_samples (int): Total number of random images to display.
        cols (int): Number of columns per row.
    
    Return:
        None
    """
    
    classes = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    
    if not classes:
        print("Error: No class folders found.")
        return

    n_rows = math.ceil(n_samples / cols)
    fig, axes = plt.subplots(n_rows, cols, figsize=(cols * 4, n_rows * 4))
    
    if n_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i in range(len(axes)):
        if i < n_samples:
            while True:
                random_class = random.choice(classes)
                class_path = os.path.join(root_path, random_class)
                images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                if images:
                    random_image_file = random.choice(images)
                    break
            
            img_path = os.path.join(class_path, random_image_file)
            
            try:
                img = Image.open(img_path)
                axes[i].imshow(img)
                axes[i].set_title(f"Class: {random_class}", fontsize=12, fontweight='bold')
                axes[i].set_xlabel(f"Shape: {img.size}", fontsize=12)
                axes[i].set_xticks([])
                axes[i].set_yticks([])
                
            except Exception as e:
                print(f"Could not load {img_path}: {e}")
        else:
            axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def unnormalize(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return img * std + mean

def data_verification(dataset: torch.utils.data.Dataset,
                      class_names: list,
                      n_rows: int =2,
                      n_cols: int =5,
                      figsize: tuple =(12,6)):
    """
    Visualizing a batch of training data to verify input shapes, label correctness,
    and augmentation intensity. This step confirms the data pipeline is ready for training.
    
    Args:
        dataset: The PyTorch dataset to visualize (must support indexing).
        class_names: A list of string labels corresponding to the class indices (e.g., ['call', 'ok']).
        n_rows: Number of rows in the visualization grid. Defaults to 2.
        n_cols: Number of columns in the visualization grid. Defaults to 5.
        figsize: The dimensions of the matplotlib figure (width, height). Defaults to (12, 6).

    Returns:
        None: Displays the plot directly.
    """
    plt.figure(figsize=figsize)

    total_images = n_rows * n_cols
    indices = random.sample(range(len(dataset)), total_images)

    for i, idx in enumerate(indices):
        img, label = dataset[idx]

        img = unnormalize(img)
        img = img.permute(1, 2, 0).numpy()

        ax = plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(img)
        plt.title(class_names[label])
        plt.axis("off")

    plt.tight_layout();

def evaluate_best_model(model: torch.nn.Module,
                        loss_fn: torch.nn.Module,
                        dataloader: torch.utils.data.DataLoader,
                        class_names: List[str],
                        device: torch.device = 'cpu'):
    """
    Evaluates a trained model on a specific dataset, plots the confusion matrix,
    and prints key performance metrics.

    Args:
        model: The trained PyTorch model to evaluate.
        loss_fn: The loss function (e.g., CrossEntropy).
        dataloader: DataLoader containing the test data.
        class_names: A list of class strings (e.g., ['ok', 'peace',etc..]).
        device: The target device (e.g., 'cuda', 'cpu').Defaults to 'cpu'.

    Returns:
        None: This function displays a plot and prints metrics to the console.

    Example:
        best_model_evaluation(model=model,
                   loss_fn=loss_fn,
                   dataloader=test_dataloader,
                   class_names=['ok', 'peace',etc..],
                   device=device)
    """
    
    # 1. Setup Multi-Class Metrics
    num_classes = len(class_names)
    
    metrics = MetricCollection({
        'acc': Accuracy(task="multiclass", num_classes=num_classes),
        'prec': Precision(task="multiclass", num_classes=num_classes, average='macro'),
        'rec': Recall(task="multiclass", num_classes=num_classes, average='macro'),
        'f1': F1Score(task="multiclass", num_classes=num_classes, average='macro'),
        'auc': AUROC(task="multiclass", num_classes=num_classes) 
    }).to(device)

    print("Running final evaluation...")
    
    # 2. Run the Test Step
    test_loss, test_res, y_preds, y_targets, y_probs  = test_step(
        dataloader=dataloader,
        model=model,
        loss_fn=loss_fn,
        metrics=metrics,
        device=device
    )
    y_pred_np = y_preds.cpu().numpy()
    y_true_np = y_targets.cpu().numpy()

    # 3. Print Results
    print("=" * 30)
    print("FINAL TEST METRICS")
    print("=" * 30)
    print(f"Accuracy:  {test_res['acc'].item()*100:.2f}%")
    print(f"AUC Score: {test_res['auc'].item():.4f}")
    print(f"Loss:      {test_loss:.4f}")
    print("=" * 30)
    print(classification_report(y_true_np, y_pred_np, target_names=class_names))
    print("=" * 30)

    fig, ax = plt.subplots(1, 2, figsize=(20, 8))

    # 4. Generate Confusion Matrix
    confmat_metric = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)
    confmat_tensor = confmat_metric(preds=y_preds.to(device), target=y_targets.to(device))
    cm_array = confmat_tensor.cpu().numpy()

    sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax[0])
    ax[0].set_title("Confusion Matrix", fontsize=16)
    ax[0].set_xlabel("Predicted Class")
    ax[0].set_ylabel("True Class")

    # 5. Generate ROC Curve
    roc = ROC(task="multiclass", num_classes=num_classes).to(device)
    fpr, tpr, thresholds = roc(y_probs, y_targets)

    for i in range(num_classes):
        ax[1].plot(fpr[i].cpu(), tpr[i].cpu(), label=f'{class_names[i]}', linewidth=2)

    ax[1].plot([0, 1], [0, 1], 'k--', label='Random Guess')
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')
    ax[1].set_title('ROC Curve')
    ax[1].legend(loc='lower right')
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()
