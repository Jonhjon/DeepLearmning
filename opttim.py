import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm # 用於顯示漂亮的進度條
from Model import SimpleCNN
import multiprocessing
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
import numpy as np

# --- 2. 偵測裝置 (GPU 或 CPU) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# --- 1. 使用者設定區 (請在這裡修改你的設定) ---

# ！！【請務必修改】！！
# 這是你建立的 "Corn_Leaves_Split" 資料夾的「父目錄」路徑
# 預設為 'Corn_Leaves_Split'，若該資料夾位於 `HW2/Corn_Leaves_Split`，腳本會自動嘗試尋找。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(BASE_DIR, "output_data")

# 如果上面指定的路徑找不到，嘗試幾個常見候選路徑（相對於專案根或本檔案目錄）
if not os.path.isdir(BASE_DATA_DIR):
    # 優先嘗試專案子目錄 HW2/Corn_Leaves_Split
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(script_dir, "output_data")
    if os.path.isdir(candidate):
        BASE_DATA_DIR = candidate
    else:
        # 再嘗試以當前工作目錄為基準
        candidate2 = os.path.join(os.getcwd(), BASE_DATA_DIR)
        if os.path.isdir(candidate2):
            BASE_DATA_DIR = candidate2

# --- 訓練參數 ---
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 100
NUM_CLASSES = 4        # 我們的專案固定是 4 類
LEARNING_RATE = 0.001  # 學習率
NUM_EPOCHS = 35        # 訓練回合數 (可以先從 15 開始)

# 儲存最佳模型的檔案名稱
BEST_MODEL_PATH = "./HW2/corn_leaf_best_model.pth"

# --- 3. 定義資料轉換 (Transforms) ---

# (1) 訓練集 (train) 的轉換 -> 包含資料增強
train_transforms = T.Compose([
    T.Resize((IMG_HEIGHT, IMG_WIDTH)), # 縮放
    # T.RandomHorizontalFlip(),          # 隨機水平翻轉
    # T.RandomRotation(10),              # 隨機旋轉 10 度
    T.ToTensor(),                      # 轉換為 Tensor (0.0 ~ 1.0)
    T.Normalize(mean=[0.485, 0.456, 0.406], # 標準化
                std=[0.229, 0.224, 0.225])
])

# (2) 測試集/驗證集 (test) 的轉換 -> *不*包含資料增強
test_transforms = T.Compose([
    T.Resize((IMG_HEIGHT, IMG_WIDTH)), # 只做縮放
    T.ToTensor(),                      # 轉換為 Tensor
    T.Normalize(mean=[0.485, 0.456, 0.406], # 標準化
                std=[0.229, 0.224, 0.225])
])

# 劃出訓練的損失圖
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

    # 儲存 CSV
    try:
        with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            writer.writerow(['class', 'precision', 'recall', 'f1', 'support'])
            for name, p, r, f1, s in zip(class_names, precisions, recalls, f1s, supports):
                writer.writerow([name, f"{float(p):.4f}", f"{float(r):.4f}", f"{float(f1):.4f}", int(s)])
            writer.writerow([])
            writer.writerow(['overall_accuracy', f"{acc:.4f}"])
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

def main():
    # 載入資料集
    try:
        train_loader, val_loader, test_loader, class_names = load_datasets(
            data_dir=BASE_DATA_DIR,
            train_transform=train_transforms,
            eval_transform=test_transforms,
            batch_size=BATCH_SIZE,
            num_workers=4
        )
        # 用於計算 dataset 長度
        train_dataset = train_loader.dataset
        val_dataset = val_loader.dataset
        test_dataset = test_loader.dataset
    except Exception as e:
        print(f"\n【錯誤！】載入資料時發生錯誤：{e}")
        return

    # --- 6. 建立模型、損失函數、優化器 ---

    print("\n--- 正在建立模型 ---")
    model = SimpleCNN(num_classes=NUM_CLASSES)
    model = model.to(DEVICE) # ！！將模型移動到 GPU/CPU！！
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RAdam(model.parameters(), lr=LEARNING_RATE)

    print("模型建立完畢，並已移動到 " + str(DEVICE))

    # --- 7. 執行訓練迴圈 ---

    print(f"\n--- 即將開始訓練，共 {NUM_EPOCHS} 回合 (Epochs) ---")

    # 用於儲存最好的模型
    best_val_accuracy = 0.0
    # 用於繪圖的損失記錄
    train_losses = []
    val_losses = []
    # 迴圈跑 NUM_EPOCHS 次
    for epoch in range(NUM_EPOCHS):
        
        # --- (A) 訓練階段 (Training) ---
        model.train() # ！！將模型切換到「訓練模式」！！
        running_train_loss = 0.0
        
        # 使用 tqdm 顯示進度條
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]")

        for images, labels in train_progress_bar:
            # 1. 將資料移動到 DEVICE
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # 2. 梯度歸零
            optimizer.zero_grad()
            
            # 3. 前向傳播 (預測)
            outputs = model(images)
            
            # 4. 計算損失
            loss = criterion(outputs, labels)
            
            # 5. 反向傳播 (計算梯度)
            loss.backward()
            
            # 6. 更新權重
            optimizer.step()
            
            # 累加損失
            running_train_loss += loss.item() * images.size(0)
        
        # 計算此 Epoch 的平均訓練損失
        epoch_train_loss = running_train_loss / len(train_dataset)

        # --- (B) 驗證階段 (Validation / Testing) ---
        model.eval() # ！！將模型切換到「評估模式」！！
        
        running_val_loss = 0.0
        correct_predictions = 0

        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]")

        # 關閉梯度計算，節省記憶體並加速
        with torch.no_grad():
            for images, labels in val_progress_bar:
                # 1. 將資料移動到 DEVICE
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                # 2. 前向傳播
                outputs = model(images)

                # 3. 計算損失
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * images.size(0)

                # 4. 計算準確度
                _, predicted_labels = torch.max(outputs, 1)
                correct_predictions += (predicted_labels == labels).sum().item()

        # 計算此 Epoch 的平均驗證損失和總準確度
        epoch_val_loss = running_val_loss / len(val_dataset)
        epoch_val_accuracy = correct_predictions / len(val_dataset)

        # 記錄損失用於後續繪圖
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        # --- (C) 印出結果 ---
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} 完成:")
        print(f"  [Train] Loss: {epoch_train_loss:.4f}")
        print(f"  [Val]   Loss: {epoch_val_loss:.4f} | Accuracy: {epoch_val_accuracy:.4f} ({(epoch_val_accuracy*100):.2f}%)")

        # --- (D) 儲存最好的模型 ---
        if epoch_val_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_val_accuracy
            # 儲存模型的「狀態字典」(state_dict)
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  -> 新高！驗證準確度達到 {(best_val_accuracy*100):.2f}%。模型已儲存至 {BEST_MODEL_PATH}")

    print("\n--- 訓練完成 ---")
    print(f"最好的驗證準確度為: {(best_val_accuracy*100):.2f}%")
    print(f"最好的模型已儲存在 {BEST_MODEL_PATH}")

    # 最終在測試集上評估
    print('\n--- 在測試集上做最終評估 ---')
    test_running_loss = 0.0
    test_correct = 0
    all_true = []
    all_pred = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Final Test'):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            test_correct += (preds == labels).sum().item()
            # collect for confusion matrix
            all_true.extend(labels.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())

    test_loss = test_running_loss / len(test_dataset)
    test_acc = test_correct / len(test_dataset)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    # 計算並儲存 Accuracy/Precision/Recall/F1
    try:
        compute_and_save_metrics(all_true, all_pred, class_names=class_names, out_csv='RAdam_1_metrics.csv')
    except Exception as e:
        print(f"產生評估指標時發生錯誤: {e}")

    # 繪製並儲存訓練/驗證損失曲線
    draw_train_loss(train_losses, val_losses, out_path='RAdam_1_train_val_loss.png')

    # 繪製並儲存混淆矩陣（原始與正規化）
    try:
        draw_confusion_matrix(all_true, all_pred, class_names=class_names, out_path='RAdam_1_confusion_matrix.png', normalize=False)
        draw_confusion_matrix(all_true, all_pred, class_names=class_names, out_path='RAdam_1_confusion_matrix_normalized.png', normalize=True)
    except Exception as e:
        print(f"產生混淆矩陣時發生錯誤: {e}")


if __name__ == '__main__':
    # 在 Windows 上啟動多程序前先呼叫 freeze_support
    multiprocessing.freeze_support()
    main()