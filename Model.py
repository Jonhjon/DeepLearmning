import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import torch
import torch.nn.functional as F
import numpy as np
try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except Exception:
    _HAS_MATPLOTLIB = False
try:
    from sklearn.metrics import confusion_matrix as _sk_confusion_matrix
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, input_size=32):
        super().__init__()
        # 若 input_size 更大（例如 224），模型會自適應全域池化
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), 
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
            
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
class SimpleCNN2_deeper(nn.Module):
    def __init__(self, num_classes=10, input_size=32):
        super().__init__()
        # 若 input_size 更大（例如 224），模型會自適應全域池化
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), 
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1), 
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1), 
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    

def draw_train_loss(train_losses, val_losses=None, out_path='train_loss.png'):
    """繪製並儲存訓練/驗證損失曲線圖。
    - train_losses: list of per-epoch training loss
    - val_losses: optional list of per-epoch validation loss
    - out_path: 輸出檔案路徑 (PNG)
"""
    if not _HAS_MATPLOTLIB:
        print("matplotlib 未安裝，若要產生損失圖請執行: pip install matplotlib")
        return
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    if val_losses is not None:
        plt.plot(epochs, val_losses, marker='o', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    try:
        plt.savefig(out_path)
        print(f"已儲存損失圖到: {out_path}")
    except Exception as e:
        print(f"儲存損失圖失敗: {e}")
    finally:
        plt.close()

def compute_and_save_metrics(y_true, y_pred, class_names=None, out_csv='metrics.csv'):
    """計算並儲存 Accuracy, Precision, Recall, F1-score。
    - y_true, y_pred: lists/arrays of integer labels
    - class_names: optional list of class names
    - out_csv: CSV 輸出路徑
    會同時印出摘要並將每個類別的 precision/recall/f1/support 寫入 CSV，最後一列為 overall accuracy。
    """
    import csv
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if class_names is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
        class_names = [str(l) for l in labels]
    n_classes = len(class_names)

    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        acc = float(accuracy_score(y_true, y_pred))
        # per-class (labels from 0..n_classes-1)
        precisions = precision_score(y_true, y_pred, labels=list(range(n_classes)), average=None, zero_division=0)
        recalls = recall_score(y_true, y_pred, labels=list(range(n_classes)), average=None, zero_division=0)
        f1s = f1_score(y_true, y_pred, labels=list(range(n_classes)), average=None, zero_division=0)
        supports = np.bincount(y_true, minlength=n_classes)
        # overall metrics (macro average)
        overall_precision = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
        overall_recall = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
        overall_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        overall_support = int(supports.sum())
    except Exception:
        # fallback using numpy and confusion matrix
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            try:
                cm[int(t), int(p)] += 1
            except Exception:
                pass
        supports = cm.sum(axis=1)
        tps = np.diag(cm).astype(float)
        fps = cm.sum(axis=0).astype(float) - tps
        fns = cm.sum(axis=1).astype(float) - tps
        with np.errstate(all='ignore'):
            precisions = np.divide(tps, tps + fps, out=np.zeros_like(tps), where=(tps+fps)!=0)
            recalls = np.divide(tps, tps + fns, out=np.zeros_like(tps), where=(tps+fns)!=0)
            f1s = np.divide(2 * precisions * recalls, precisions + recalls, out=np.zeros_like(precisions), where=(precisions+recalls)!=0)
        acc = float(tps.sum() / cm.sum()) if cm.sum() != 0 else 0.0
        # overall metrics (macro average)
        overall_precision = float(np.mean(precisions))
        overall_recall = float(np.mean(recalls))
        overall_f1 = float(np.mean(f1s))
        overall_support = int(supports.sum())

    # 儲存 CSV
    try:
        with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            writer.writerow(['class', 'precision', 'recall', 'f1', 'support'])
            for name, p, r, f1, s in zip(class_names, precisions, recalls, f1s, supports):
                writer.writerow([name, f"{float(p):.4f}", f"{float(r):.4f}", f"{float(f1):.4f}", int(s)])
            writer.writerow([])
            writer.writerow(['overall', f"{overall_precision:.4f}", f"{overall_recall:.4f}", f"{overall_f1:.4f}", overall_support])
            writer.writerow(['overall_accuracy', f"{acc:.4f}", '', '', ''])
        print(f"已儲存評估指標到: {out_csv}")
    except Exception as e:
        print(f"儲存評估指標失敗: {e}")

    # 印出摘要
    print('\n--- Evaluation Metrics Summary ---')
    print(f"Overall Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    for idx, (name, p, r, f1, s) in enumerate(zip(class_names, precisions, recalls, f1s, supports)):
        print(f"  Class {idx} ({name}): Precision={float(p):.4f}, Recall={float(r):.4f}, F1={float(f1):.4f}, Support={int(s)}")
    print('----------------------------------\n')
#劃出混淆矩陣
def draw_confusion_matrix(y_true, y_pred, class_names=None, out_path='confusion_matrix.png', normalize=False):
    """繪製並儲存混淆矩陣。
    - y_true, y_pred: lists/arrays of integer labels
    - class_names: list of class label names (optional)
    - out_path: 輸出 PNG 檔名
    - normalize: 是否正規化每一列 (True => 每列和為 1)
    """
    if not _HAS_MATPLOTLIB:
        print("matplotlib 未安裝，若要產生混淆矩陣請執行: pip install matplotlib")
        return

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if class_names is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
        class_names = [str(l) for l in labels]
    n_classes = len(class_names)

    if _HAS_SKLEARN:
        cm = _sk_confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    else:
        # simple numpy implementation
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            try:
                cm[int(t), int(p)] += 1
            except Exception:
                pass
    if normalize:
        with np.errstate(all='ignore'):
            row_sums = cm.sum(axis=1, keepdims=True).astype(float)
            cm_normalized = np.divide(cm, row_sums, where=(row_sums != 0))
            display = cm_normalized
    else:
        display = cm

    plt.figure(figsize=(8, 6))
    im = plt.imshow(display, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix' + (" (normalized)" if normalize else ""))
    plt.colorbar(im, fraction=0.046, pad=0.04)

    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)

    fmt = '.2f' if normalize else 'd'
    thresh = display.max() / 2. if display.size else 0
    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            val = display[i, j]
            if np.isnan(val):
                txt = '0'
            else:
                txt = format(val, fmt)
            plt.text(j, i, txt,
                    horizontalalignment='center',
                    color='white' if display[i, j] > thresh else 'black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    try:
        plt.savefig(out_path)
        print(f"已儲存混淆矩陣到: {out_path}")
    except Exception as e:
        print(f"儲存混淆矩陣失敗: {e}")
    finally:
        plt.close()

# --- 4. 載入 Datasets 和 DataLoaders ---
def load_datasets(data_dir, train_transform, eval_transform, batch_size=32, num_workers=4):
    """
    載入圖片資料集並建立 DataLoaders。
    
    參數:
        data_dir (str): 資料夾根目錄路徑
        train_transform: 訓練集的轉換
        eval_transform: 驗證/測試集的轉換
        batch_size (int): 批次大小
        num_workers (int): 資料載入的執行緒數
    
    回傳:
        tuple: (train_loader, val_loader, test_loader, class_names)
    """
    print("\n--- 正在載入資料 ---")
    print(f"資料目錄: {data_dir}")
    
    # 定義 train、val、test 資料夾路徑
    TRAIN_DIR = os.path.join(data_dir, "train")
    VAL_DIR = os.path.join(data_dir, "val")
    TEST_DIR = os.path.join(data_dir, "test")
    
    # 檢查資料夾是否存在
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"找不到資料夾 '{data_dir}'。請確認路徑是否正確。")
    
    if not (os.path.isdir(TRAIN_DIR) and os.path.isdir(VAL_DIR) and os.path.isdir(TEST_DIR)):
        raise FileNotFoundError(
            f"資料夾結構不正確。需要 train/、val/、test/ 三個子資料夾。\n"
            f"找到的路徑: {data_dir}\n"
            f"- train 存在: {os.path.isdir(TRAIN_DIR)}\n"
            f"- val 存在: {os.path.isdir(VAL_DIR)}\n"
            f"- test 存在: {os.path.isdir(TEST_DIR)}"
        )
    
    # 使用 ImageFolder 載入資料集
    train_dataset = ImageFolder(root=TRAIN_DIR, transform=train_transform)
    val_dataset = ImageFolder(root=VAL_DIR, transform=eval_transform)
    test_dataset = ImageFolder(root=TEST_DIR, transform=eval_transform)
    
    # 驗證資料集是否為空
    if len(train_dataset) == 0:
        raise RuntimeError(f"訓練集 {TRAIN_DIR} 內沒有找到任何影像檔。請確認路徑與檔案格式。")
    if len(val_dataset) == 0:
        raise RuntimeError(f"驗證集 {VAL_DIR} 內沒有找到任何影像檔。請確認路徑與檔案格式。")
    if len(test_dataset) == 0:
        raise RuntimeError(f"測試集 {TEST_DIR} 內沒有找到任何影像檔。請確認路徑與檔案格式。")
    
    # 建立 DataLoaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 取得類別名稱
    class_names = train_dataset.classes
    
    # 印出資訊
    print("資料集載入成功！")
    print(f"訓練集圖片數量: {len(train_dataset)}")
    print(f"驗證集圖片數量: {len(val_dataset)}")
    print(f"測試集圖片數量: {len(test_dataset)}")
    print(f"偵測到 {len(class_names)} 個類別: {class_names}")
    
    return train_loader, val_loader, test_loader, class_names