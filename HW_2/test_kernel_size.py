"""
測試不同 kernel size 對模型訓練的影響
使用 SimpleCNN 和 SimpleCNN2_deeper 模型
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
import multiprocessing
import numpy as np

# 導入自定義模型和函式
from Model import load_datasets, draw_train_loss, compute_and_save_metrics, draw_confusion_matrix

# --- 偵測裝置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {DEVICE}")

# --- 設定資料路徑 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(BASE_DIR, "output_data_augmented")

if not os.path.isdir(BASE_DATA_DIR):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(script_dir, "output_data_augmented")
    if os.path.isdir(candidate):
        BASE_DATA_DIR = candidate
    else:
        candidate2 = os.path.join(os.getcwd(), "HW2", "output_data_augmented")
        if os.path.isdir(candidate2):
            BASE_DATA_DIR = candidate2

# --- 訓練參數 ---
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 64
NUM_CLASSES = 4
LEARNING_RATE = 0.001
NUM_EPOCHS = 40

# 輸出目錄
OUTPUT_DIR = "./HW2/different_Kernal_size"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 測試的 kernel sizes
KERNEL_SIZES = [3, 5, 7]

# --- 資料轉換 ---
train_transforms = T.Compose([
    # T.Resize((IMG_HEIGHT, IMG_WIDTH)),
    # T.RandomHorizontalFlip(p=0.5),
    # T.RandomRotation(15),
    # T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = T.Compose([
    # T.Resize((IMG_HEIGHT, IMG_WIDTH)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# === 定義可調整 kernel_size 的模型 ===

class SimpleCNN_Variable(nn.Module):
    def __init__(self, num_classes=4, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2  # 保持輸出大小
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=kernel_size, padding=padding), 
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding), 
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

class SimpleCNN2_deeper_Variable(nn.Module):
    def __init__(self, num_classes=4, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2  # 保持輸出大小
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=kernel_size, padding=padding), 
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding), 
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=kernel_size, padding=padding), 
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=kernel_size, padding=padding), 
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

def train_model(model, model_name, kernel_size, train_loader, val_loader, test_loader, class_names):
    """訓練單個模型"""
    print(f"\n{'='*70}")
    print(f"訓練模型: {model_name} | Kernel Size: {kernel_size}x{kernel_size}")
    print(f"{'='*70}\n")
    
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    test_dataset = test_loader.dataset
    
    best_val_accuracy = 0.0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # 訓練迴圈
    for epoch in range(NUM_EPOCHS):
        # === 訓練階段 ===
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        
        for images, labels in train_pbar:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        epoch_train_loss = running_train_loss / len(train_dataset)
        epoch_train_acc = train_correct / len(train_dataset)
        
        # === 驗證階段 ===
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
        
        epoch_val_loss = running_val_loss / len(val_dataset)
        epoch_val_acc = val_correct / len(val_dataset)
        
        # 記錄
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accuracies.append(epoch_train_acc)
        val_accuracies.append(epoch_val_acc)
        
        # 調整學習率
        scheduler.step(epoch_val_loss)
        
        # 顯示結果
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"  [Train] Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.4f} ({epoch_train_acc*100:.2f}%)")
        print(f"  [Val]   Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc:.4f} ({epoch_val_acc*100:.2f}%)")
        
        # 儲存最佳模型
        if epoch_val_acc > best_val_accuracy:
            best_val_accuracy = epoch_val_acc
            model_path = os.path.join(OUTPUT_DIR, f"{model_name}_k{kernel_size}_best.pth")
            torch.save(model.state_dict(), model_path)
            print(f"  ✓ 新高！已儲存模型 (Val Acc: {best_val_accuracy*100:.2f}%)")
    
    # === 測試階段 ===
    print(f"\n載入最佳模型進行測試...")
    model_path = os.path.join(OUTPUT_DIR, f"{model_name}_k{kernel_size}_best.pth")
    model.load_state_dict(torch.load(model_path, weights_only=False))
    model.eval()
    
    test_correct = 0
    all_true = []
    all_pred = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            test_correct += (preds == labels).sum().item()
            
            all_true.extend(labels.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())
    
    test_acc = test_correct / len(test_dataset)
    
    print(f"\n測試結果:")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # === 儲存結果 ===
    prefix = f"{model_name}_k{kernel_size}"
    
    # 儲存指標
    compute_and_save_metrics(
        all_true, all_pred, 
        class_names=class_names, 
        out_csv=os.path.join(OUTPUT_DIR, f'{prefix}_metrics.csv')
    )
    
    # 繪製損失曲線
    draw_train_loss(
        train_losses, val_losses, 
        out_path=os.path.join(OUTPUT_DIR, f'{prefix}_loss.png')
    )
    
    # 繪製混淆矩陣
    draw_confusion_matrix(
        all_true, all_pred, 
        class_names=class_names, 
        out_path=os.path.join(OUTPUT_DIR, f'{prefix}_confusion_matrix.png'), 
        normalize=False
    )
    draw_confusion_matrix(
        all_true, all_pred, 
        class_names=class_names, 
        out_path=os.path.join(OUTPUT_DIR, f'{prefix}_confusion_matrix_normalized.png'), 
        normalize=True
    )
    
    return {
        'model_name': model_name,
        'kernel_size': kernel_size,
        'best_val_acc': best_val_accuracy,
        'test_acc': test_acc,
        'train_losses': train_losses,
        'val_losses': val_losses
    }

def main():
    
    print(f"Using device: {DEVICE}")

    print(f"\n{'='*70}")
    print("測試不同 Kernel Size 對模型訓練的影響")
    print(f"{'='*70}")
    print(f"測試 Kernel Sizes: {KERNEL_SIZES}")
    print(f"模型: SimpleCNN, SimpleCNN2_deeper")
    print(f"輸出目錄: {OUTPUT_DIR}")
    print(f"{'='*70}\n")
    
    # 載入資料集 (只需載入一次)
    try:
        train_loader, val_loader, test_loader, class_names = load_datasets(
            data_dir=BASE_DATA_DIR,
            train_transform=train_transforms,
            eval_transform=test_transforms,
            batch_size=BATCH_SIZE,
            num_workers=4
        )
    except Exception as e:
        print(f"載入資料時發生錯誤：{e}")
        return
    
    results = []
    
    # 測試 SimpleCNN 模型
    for kernel_size in KERNEL_SIZES:
        model = SimpleCNN_Variable(num_classes=NUM_CLASSES, kernel_size=kernel_size)
        result = train_model(
            model, "SimpleCNN", kernel_size,
            train_loader, val_loader, test_loader, class_names
        )
        results.append(result)
    
    # 測試 SimpleCNN2_deeper 模型
    for kernel_size in KERNEL_SIZES:
        model = SimpleCNN2_deeper_Variable(num_classes=NUM_CLASSES, kernel_size=kernel_size)
        result = train_model(
            model, "SimpleCNN2_deeper", kernel_size,
            train_loader, val_loader, test_loader, class_names
        )
        results.append(result)
    
    # === 生成總結報告 ===
    print(f"\n{'='*70}")
    print("實驗結果總結")
    print(f"{'='*70}\n")
    
    # 儲存為 TXT 格式
    # summary_txt_path = os.path.join(OUTPUT_DIR, "summary.txt")
    # with open(summary_txt_path, 'w', encoding='utf-8') as f:
    #     f.write("=" * 70 + "\n")
    #     f.write("Kernel Size 實驗結果總結\n")
    #     f.write("=" * 70 + "\n\n")
        
    #     for result in results:
    #         summary = (
    #             f"模型: {result['model_name']}\n"
    #             f"Kernel Size: {result['kernel_size']}x{result['kernel_size']}\n"
    #             f"最佳驗證準確度: {result['best_val_acc']:.4f} ({result['best_val_acc']*100:.2f}%)\n"
    #             f"測試準確度: {result['test_acc']:.4f} ({result['test_acc']*100:.2f}%)\n"
    #             f"{'-'*70}\n"
    #         )
    #         print(summary)
    #         f.write(summary + "\n")
    
    # 儲存為 CSV 格式
    import csv
    summary_csv_path = os.path.join(OUTPUT_DIR, "summary.csv")
    with open(summary_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Kernel_Size', 'Best_Val_Accuracy', 'Test_Accuracy', 'Best_Val_Acc_%', 'Test_Acc_%'])
        
        for result in results:
            writer.writerow([
                result['model_name'],
                f"{result['kernel_size']}x{result['kernel_size']}",
                f"{result['best_val_acc']:.4f}",
                f"{result['test_acc']:.4f}",
                f"{result['best_val_acc']*100:.2f}%",
                f"{result['test_acc']*100:.2f}%"
            ])
    
    print(f"總結報告已儲存至:")
    # print(f"  - TXT: {summary_txt_path}")
    print(f"  - CSV: {summary_csv_path}")
    print(f"\n所有實驗結果已儲存至: {OUTPUT_DIR}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
