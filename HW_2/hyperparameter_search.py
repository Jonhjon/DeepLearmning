"""
超參數搜尋: kernel_size, padding, stride
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
import csv
from itertools import product

# 導入自定義模型和函式
from Model import load_datasets, draw_train_loss, compute_and_save_metrics, draw_confusion_matrix

# --- 偵測裝置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

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
NUM_EPOCHS = 25  # 超參數搜尋時可以用較少 epochs

# 輸出目錄
OUTPUT_DIR = "./HW2/hyperparameter_search"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 超參數搜尋空間 ===
KERNEL_SIZES = [3, 5, 7]
PADDINGS = [0, 1, 2, 3]  # 0=valid, 1-3=same with different sizes
STRIDES = [1, 2]  # 1=normal, 2=替代MaxPool

# --- 資料轉換 ---
train_transforms = T.Compose([
    T.Resize((IMG_HEIGHT, IMG_WIDTH)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = T.Compose([
    T.Resize((IMG_HEIGHT, IMG_WIDTH)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# === 定義可調整超參數的模型 ===

class SimpleCNN_Flexible(nn.Module):
    def __init__(self, num_classes=4, kernel_size=3, padding=1, stride=1):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=kernel_size, padding=padding, stride=stride), 
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding, stride=stride), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding, stride=stride), 
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


class SimpleCNN2_deeper_Flexible(nn.Module):
    def __init__(self, num_classes=4, kernel_size=3, padding=1, stride=1):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=kernel_size, padding=padding, stride=stride), 
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding, stride=stride), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding, stride=stride), 
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=kernel_size, padding=padding, stride=stride), 
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=kernel_size, padding=padding, stride=stride), 
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


def train_model(model, config, train_loader, val_loader, test_loader, class_names):
    """訓練單個模型配置"""
    model_name = config['model_name']
    kernel_size = config['kernel_size']
    padding = config['padding']
    stride = config['stride']
    
    config_str = f"k{kernel_size}_p{padding}_s{stride}"
    
    print(f"\n{'='*70}")
    print(f"訓練: {model_name} | Kernel={kernel_size}, Padding={padding}, Stride={stride}")
    print(f"{'='*70}\n")
    
    try:
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
        
        # 訓練迴圈
        for epoch in range(NUM_EPOCHS):
            # === 訓練階段 ===
            model.train()
            running_train_loss = 0.0
            train_correct = 0
            
            for images, labels in train_loader:
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
            
            # 調整學習率
            scheduler.step(epoch_val_loss)
            
            # 簡短輸出
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS}: Train Acc={epoch_train_acc:.4f}, Val Acc={epoch_val_acc:.4f}")
            
            # 儲存最佳模型
            if epoch_val_acc > best_val_accuracy:
                best_val_accuracy = epoch_val_acc
        
        # === 測試階段 ===
        model.eval()
        test_correct = 0
        all_true = []
        all_pred = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                test_correct += (preds == labels).sum().item()
                
                all_true.extend(labels.cpu().tolist())
                all_pred.extend(preds.cpu().tolist())
        
        test_acc = test_correct / len(test_dataset)
        
        print(f"✓ 完成 | Val Acc: {best_val_accuracy:.4f}, Test Acc: {test_acc:.4f}\n")
        
        return {
            'model_name': model_name,
            'kernel_size': kernel_size,
            'padding': padding,
            'stride': stride,
            'best_val_acc': best_val_accuracy,
            'test_acc': test_acc,
            'success': True
        }
        
    except Exception as e:
        print(f"✗ 錯誤: {e}\n")
        return {
            'model_name': model_name,
            'kernel_size': kernel_size,
            'padding': padding,
            'stride': stride,
            'best_val_acc': 0.0,
            'test_acc': 0.0,
            'success': False,
            'error': str(e)
        }


def main():
    print(f"\n{'='*70}")
    print("超參數搜尋: Kernel Size, Padding, Stride")
    print(f"{'='*70}")
    print(f"Kernel Sizes: {KERNEL_SIZES}")
    print(f"Paddings: {PADDINGS}")
    print(f"Strides: {STRIDES}")
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
    
    # 生成所有超參數組合
    hyperparameter_combinations = list(product(KERNEL_SIZES, PADDINGS, STRIDES))
    total_experiments = len(hyperparameter_combinations) * 2  # 2個模型
    
    print(f"總共需要執行 {total_experiments} 個實驗")
    print(f"預估時間: 約 {total_experiments * NUM_EPOCHS * 2} 分鐘\n")
    
    experiment_count = 0
    
    # 測試 SimpleCNN 模型
    print(f"\n{'#'*70}")
    print(f"# 開始測試 SimpleCNN 模型")
    print(f"{'#'*70}\n")
    
    for kernel_size, padding, stride in hyperparameter_combinations:
        experiment_count += 1
        print(f"[{experiment_count}/{total_experiments}] ", end="")
        
        model = SimpleCNN_Flexible(
            num_classes=NUM_CLASSES, 
            kernel_size=kernel_size,
            padding=padding,
            stride=stride
        )
        
        config = {
            'model_name': 'SimpleCNN',
            'kernel_size': kernel_size,
            'padding': padding,
            'stride': stride
        }
        
        result = train_model(model, config, train_loader, val_loader, test_loader, class_names)
        results.append(result)
        
        # 即時儲存結果
        save_intermediate_results(results)
    
    # 測試 SimpleCNN2_deeper 模型
    print(f"\n{'#'*70}")
    print(f"# 開始測試 SimpleCNN2_deeper 模型")
    print(f"{'#'*70}\n")
    
    for kernel_size, padding, stride in hyperparameter_combinations:
        experiment_count += 1
        print(f"[{experiment_count}/{total_experiments}] ", end="")
        
        model = SimpleCNN2_deeper_Flexible(
            num_classes=NUM_CLASSES, 
            kernel_size=kernel_size,
            padding=padding,
            stride=stride
        )
        
        config = {
            'model_name': 'SimpleCNN2_deeper',
            'kernel_size': kernel_size,
            'padding': padding,
            'stride': stride
        }
        
        result = train_model(model, config, train_loader, val_loader, test_loader, class_names)
        results.append(result)
        
        # 即時儲存結果
        save_intermediate_results(results)
    
    # === 生成最終報告 ===
    generate_final_report(results)


def save_intermediate_results(results):
    """即時儲存中間結果到CSV"""
    csv_path = os.path.join(OUTPUT_DIR, "hyperparameter_results.csv")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Model', 'Kernel_Size', 'Padding', 'Stride', 
            'Best_Val_Accuracy', 'Test_Accuracy', 
            'Val_Acc_%', 'Test_Acc_%', 'Success', 'Error'
        ])
        
        for r in results:
            writer.writerow([
                r['model_name'],
                r['kernel_size'],
                r['padding'],
                r['stride'],
                f"{r['best_val_acc']:.4f}",
                f"{r['test_acc']:.4f}",
                f"{r['best_val_acc']*100:.2f}%",
                f"{r['test_acc']*100:.2f}%",
                r['success'],
                r.get('error', '')
            ])


def generate_final_report(results):
    """生成最終報告"""
    print(f"\n{'='*70}")
    print("實驗結果總結")
    print(f"{'='*70}\n")
    
    # 過濾成功的結果
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("沒有成功的實驗結果！")
        return
    
    # 按驗證準確度排序
    successful_results.sort(key=lambda x: x['best_val_acc'], reverse=True)
    
    # 儲存完整結果 CSV
    save_intermediate_results(results)
    
    # 生成 Top 10 報告
    print("\n=== Top 10 最佳配置 (按驗證準確度) ===\n")
    top_10_path = os.path.join(OUTPUT_DIR, "top_10_configs.csv")
    
    with open(top_10_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Rank', 'Model', 'Kernel_Size', 'Padding', 'Stride', 
            'Val_Accuracy', 'Test_Accuracy'
        ])
        
        for i, r in enumerate(successful_results[:10], 1):
            print(f"{i}. {r['model_name']} | k={r['kernel_size']}, p={r['padding']}, s={r['stride']} | "
                  f"Val: {r['best_val_acc']:.4f} ({r['best_val_acc']*100:.2f}%) | "
                  f"Test: {r['test_acc']:.4f} ({r['test_acc']*100:.2f}%)")
            
            writer.writerow([
                i,
                r['model_name'],
                r['kernel_size'],
                r['padding'],
                r['stride'],
                f"{r['best_val_acc']:.4f}",
                f"{r['test_acc']:.4f}"
            ])
    
    # 最佳配置
    best = successful_results[0]
    print(f"\n{'='*70}")
    print("最佳配置")
    print(f"{'='*70}")
    print(f"模型: {best['model_name']}")
    print(f"Kernel Size: {best['kernel_size']}")
    print(f"Padding: {best['padding']}")
    print(f"Stride: {best['stride']}")
    print(f"驗證準確度: {best['best_val_acc']:.4f} ({best['best_val_acc']*100:.2f}%)")
    print(f"測試準確度: {best['test_acc']:.4f} ({best['test_acc']*100:.2f}%)")
    print(f"{'='*70}\n")
    
    # 統計分析
    print("\n=== 統計分析 ===\n")
    
    # 按 kernel size 分組
    kernel_stats = {}
    for k in KERNEL_SIZES:
        k_results = [r for r in successful_results if r['kernel_size'] == k]
        if k_results:
            avg_val = np.mean([r['best_val_acc'] for r in k_results])
            kernel_stats[k] = avg_val
            print(f"Kernel Size {k}: 平均驗證準確度 = {avg_val:.4f}")
    
    # 按 padding 分組
    print()
    padding_stats = {}
    for p in PADDINGS:
        p_results = [r for r in successful_results if r['padding'] == p]
        if p_results:
            avg_val = np.mean([r['best_val_acc'] for r in p_results])
            padding_stats[p] = avg_val
            print(f"Padding {p}: 平均驗證準確度 = {avg_val:.4f}")
    
    # 按 stride 分組
    print()
    stride_stats = {}
    for s in STRIDES:
        s_results = [r for r in successful_results if r['stride'] == s]
        if s_results:
            avg_val = np.mean([r['best_val_acc'] for r in s_results])
            stride_stats[s] = avg_val
            print(f"Stride {s}: 平均驗證準確度 = {avg_val:.4f}")
    
    print(f"\n所有結果已儲存至:")
    print(f"  - 完整結果: {OUTPUT_DIR}/hyperparameter_results.csv")
    print(f"  - Top 10: {OUTPUT_DIR}/top_10_configs.csv")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
