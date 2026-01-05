"""
訓練基礎 RNN/LSTM/GRU 模型的腳本
使用 Modle.py 中的 FeatureExtractor 提取特徵後，再輸入時序模型
使用 GetDataLoader.py 載入資料
"""
import json
import os
import itertools
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 從 Modle.py 匯入 FeatureExtractor
from Modle import FeatureExtractor
# 從 GetDataLoader.py 匯入資料載入器
from GetDataLoader import GetDataLoader


# ============================================================
# 實驗配置
# ============================================================
EXPERIMENT_CONFIGS = {
    'MODEL_TYPES': ['rnn', 'lstm', 'gru'],  # 三種基礎模型
    'HIDDEN_SIZES': [256],  # 隱藏層大小
    'NUM_LAYERS': [2],  # RNN/LSTM/GRU 層數
    'DROPOUT_RATES': [0.3],  # Dropout 率
}

# 特徵提取器配置
# FEATURE_EXTRACTOR_NAME = 'IDEA-CCNL/Erlangshen-RoBERTa-110M-Sentiment'
FEATURE_EXTRACTOR_NAME = 'intfloat/multilingual-e5-large'  # 560M 參數, 多語言模型
# 從 key.json 讀取 API token
with open('HW3/key.json', 'r') as f:
    keys = json.load(f)
USE_AUTH_TOKEN = keys.get('HW3_Key')

POOLING_METHOD = 'cls'  # 'cls', 'mean', 'last'

# 訓練配置
BATCH_SIZE = 64
FEATURE_BATCH_SIZE = 128  # 特徵提取時的批次大小
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50

# 數據路徑
TRAIN_DATA_PATH = r'C:\Users\H514 #4856\Desktop\deep_learning_114206103\HW3_Data\online_shopping\train_data.csv'
TEST_DATA_PATH = r'C:\Users\H514 #4856\Desktop\deep_learning_114206103\HW3_Data\online_shopping\test_data.csv'

# 輸出目錄
OUTPUT_DIR = 'HW3/basic_models_with_features'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 設備
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用設備: {device}")


# ============================================================
# 時序模型定義 (接收預提取的特徵)
# ============================================================
class FeatureBasedRNN(nn.Module):
    """基於預提取特徵的 RNN 分類器"""
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.3):
        super(FeatureBasedRNN, self).__init__()
        
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 雙向所以 *2
    
    def forward(self, features):
        # features: [batch, 1, input_size] (單一時間步)
        output, hidden = self.rnn(features)
        
        # 取最後一個時間步
        output = output[:, -1, :]
        output = self.dropout(output)
        logits = self.fc(output)
        return logits


class FeatureBasedLSTM(nn.Module):
    """基於預提取特徵的 LSTM 分類器"""
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.3):
        super(FeatureBasedLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, features):
        output, (hidden, cell) = self.lstm(features)
        
        # 取最後一個時間步
        output = output[:, -1, :]
        output = self.dropout(output)
        logits = self.fc(output)
        return logits


class FeatureBasedGRU(nn.Module):
    """基於預提取特徵的 GRU 分類器"""
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.3):
        super(FeatureBasedGRU, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, features):
        output, hidden = self.gru(features)
        
        # 取最後一個時間步
        output = output[:, -1, :]
        output = self.dropout(output)
        logits = self.fc(output)
        return logits


# ============================================================
# 特徵數據集類
# ============================================================
class FeatureDataset(Dataset):
    """預提取特徵的數據集"""
    
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'label': self.labels[idx]
        }


# ============================================================
# 特徵提取函數
# ============================================================
def extract_all_features(feature_extractor, dataloader, device, desc="提取特徵"):
    """
    使用 Modle.py 中的 FeatureExtractor 提取所有樣本的特徵
    返回: features [N, feature_dim], labels [N]
    """
    feature_extractor.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            # 使用 FeatureExtractor 提取特徵
            features = feature_extractor(input_ids, attention_mask)
            
            all_features.append(features.cpu())
            all_labels.append(labels)
    
    # 合併所有批次
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return all_features, all_labels


# ============================================================
# 訓練與評估
# ============================================================
def train_epoch(model, train_loader, criterion, optimizer, device):
    """訓練一個 epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in train_loader:
        # features 需要擴展維度: [batch, feature_dim] -> [batch, 1, feature_dim]
        features = batch['features'].unsqueeze(1).to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def evaluate(model, val_loader, criterion, device):
    """評估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            features = batch['features'].unsqueeze(1).to(device)
            labels = batch['label'].to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, precision, recall, f1, all_preds, all_labels


def plot_training_curves(train_losses, train_accs, val_losses, val_accs, save_path):
    """繪製訓練曲線"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_losses, label='Train Loss', marker='o', markersize=2)
    ax1.plot(val_losses, label='Val Loss', marker='s', markersize=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(train_accs, label='Train Acc', marker='o', markersize=2)
    ax2.plot(val_accs, label='Val Acc', marker='s', markersize=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(labels, preds, save_path):
    """繪製混淆矩陣"""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# 主訓練流程
# ============================================================
def run_single_experiment(model_type, hidden_size, num_layers, dropout_rate,
                          train_loader, val_loader, test_loader, input_size,
                          experiment_id, total_experiments):
    """運行單個實驗"""
    
    print("\n" + "="*60)
    print(f"實驗 {experiment_id}/{total_experiments}")
    print("="*60)
    print(f"模型類型: {model_type.upper()}")
    print(f"特徵提取器: {FEATURE_EXTRACTOR_NAME}")
    print(f"池化方式: {POOLING_METHOD}")
    print(f"輸入特徵維度: {input_size}")
    print(f"隱藏層大小: {hidden_size}")
    print(f"層數: {num_layers}")
    print(f"Dropout: {dropout_rate}")
    print("="*60)
    
    # 創建模型
    if model_type == 'rnn':
        model = FeatureBasedRNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=2,
            dropout_rate=dropout_rate
        )
    elif model_type == 'lstm':
        model = FeatureBasedLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=2,
            dropout_rate=dropout_rate
        )
    elif model_type == 'gru':
        model = FeatureBasedGRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=2,
            dropout_rate=dropout_rate
        )
    else:
        raise ValueError(f"未知的模型類型: {model_type}")
    
    model.to(device)
    
    # 計算參數量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"總參數量: {total_params:,}")
    print(f"可訓練參數量: {trainable_params:,}")
    
    # 損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 訓練歷史
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0
    best_epoch = 0
    
    # 創建實驗目錄
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"{model_type}_{timestamp}"
    model_save_dir = os.path.join(OUTPUT_DIR, exp_name)
    os.makedirs(model_save_dir, exist_ok=True)
    
    # 訓練循環
    start_time = datetime.now()
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 40)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        val_loss, val_acc, val_precision, val_recall, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Val Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(model_save_dir, 'best_model.pth'))
    
    training_time = (datetime.now() - start_time).total_seconds() / 60
    
    # 載入最佳模型進行測試
    model.load_state_dict(torch.load(os.path.join(model_save_dir, 'best_model.pth')))
    
    # 測試集評估
    test_loss, test_acc, test_precision, test_recall, test_f1, test_preds, test_labels_list = evaluate(
        model, test_loader, criterion, device
    )
    
    print("\n" + "="*60)
    print("測試集結果 (最佳模型):")
    print(f"最佳 Epoch: {best_epoch}")
    print(f"測試準確率: {test_acc:.4f}")
    print(f"測試精確率: {test_precision:.4f}")
    print(f"測試召回率: {test_recall:.4f}")
    print(f"測試 F1 分數: {test_f1:.4f}")
    print(f"訓練時間: {training_time:.2f} 分鐘")
    print("="*60)
    
    # 繪製圖表
    plot_training_curves(
        train_losses, train_accs, val_losses, val_accs,
        os.path.join(model_save_dir, 'training_curves.png')
    )
    plot_confusion_matrix(
        test_labels_list, test_preds,
        os.path.join(model_save_dir, 'confusion_matrix_test.png')
    )
    
    return {
        '實驗時間': timestamp,
        '實驗資料夾': model_save_dir,
        '特徵提取器': FEATURE_EXTRACTOR_NAME,
        '池化方式': POOLING_METHOD,
        '模型類型': model_type.upper(),
        '輸入特徵維度': input_size,
        '隱藏層大小': hidden_size,
        '層數': num_layers,
        'Dropout率': dropout_rate,
        '學習率': LEARNING_RATE,
        '批次大小': BATCH_SIZE,
        '總參數量': total_params,
        '可訓練參數量': trainable_params,
        '最佳Epoch': best_epoch,
        '測試準確率': f'{test_acc:.4f}',
        '測試精確率': f'{test_precision:.4f}',
        '測試召回率': f'{test_recall:.4f}',
        '測試F1分數': f'{test_f1:.4f}',
        '訓練時間(分鐘)': f'{training_time:.2f}'
    }


def run_all_experiments():
    """運行所有實驗"""
    print("\n" + "="*60)
    print("基礎 RNN/LSTM/GRU 模型訓練")
    print("使用 Modle.py 中的 FeatureExtractor 進行特徵提取")
    print("使用 GetDataLoader.py 載入資料")
    print("="*60)
    
    # 1. 使用 GetDataLoader 載入資料
    print(f"\n[1/4] 使用 GetDataLoader 載入資料...")
    print(f"特徵提取器: {FEATURE_EXTRACTOR_NAME}")
    
    data_loader = GetDataLoader(FEATURE_EXTRACTOR_NAME, use_auth_token=USE_AUTH_TOKEN)
    train_dataloader, val_dataloader, test_dataloader = data_loader.get_DataLoader(
        data_sum=-1,
        batch_size=FEATURE_BATCH_SIZE,
        train_path=TRAIN_DATA_PATH,
        test_path=TEST_DATA_PATH
    )
    
    print(f"訓練集批次數: {len(train_dataloader)}")
    print(f"驗證集批次數: {len(val_dataloader)}")
    print(f"測試集批次數: {len(test_dataloader)}")
    
    # 2. 初始化 FeatureExtractor
    print(f"\n[2/4] 初始化 FeatureExtractor (池化方式: {POOLING_METHOD})...")
    feature_extractor = FeatureExtractor(
        FEATURE_EXTRACTOR_NAME, 
        use_auth_token=USE_AUTH_TOKEN,
        pooling=POOLING_METHOD
    )
    feature_extractor.to(device)
    
    input_size = feature_extractor.model.config.hidden_size
    print(f"特徵維度: {input_size}")
    
    # 3. 提取所有特徵 (只需提取一次)
    print("\n[3/4] 提取特徵...")
    
    print("\n提取訓練集特徵...")
    train_features, train_labels_tensor = extract_all_features(
        feature_extractor, train_dataloader, device, "訓練集"
    )
    print(f"訓練集特徵形狀: {train_features.shape}")
    
    print("\n提取驗證集特徵...")
    val_features, val_labels_tensor = extract_all_features(
        feature_extractor, val_dataloader, device, "驗證集"
    )
    print(f"驗證集特徵形狀: {val_features.shape}")
    
    print("\n提取測試集特徵...")
    test_features, test_labels_tensor = extract_all_features(
        feature_extractor, test_dataloader, device, "測試集"
    )
    print(f"測試集特徵形狀: {test_features.shape}")
    
    # 釋放特徵提取器的 GPU 記憶體
    del feature_extractor
    torch.cuda.empty_cache()
    print("\n已釋放特徵提取器的 GPU 記憶體")
    
    # 創建特徵數據集
    train_feature_dataset = FeatureDataset(train_features, train_labels_tensor)
    val_feature_dataset = FeatureDataset(val_features, val_labels_tensor)
    test_feature_dataset = FeatureDataset(test_features, test_labels_tensor)
    
    train_loader = DataLoader(train_feature_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_feature_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_feature_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 4. 運行所有實驗
    print("\n[4/4] 開始訓練時序模型...")
    
    configs = list(itertools.product(
        EXPERIMENT_CONFIGS['MODEL_TYPES'],
        EXPERIMENT_CONFIGS['HIDDEN_SIZES'],
        EXPERIMENT_CONFIGS['NUM_LAYERS'],
        EXPERIMENT_CONFIGS['DROPOUT_RATES']
    ))
    
    total_experiments = len(configs)
    print(f"\n總共 {total_experiments} 個實驗")
    
    results = []
    for idx, (model_type, hidden_size, num_layers, dropout_rate) in enumerate(configs, 1):
        result = run_single_experiment(
            model_type, hidden_size, num_layers, dropout_rate,
            train_loader, val_loader, test_loader, input_size,
            idx, total_experiments
        )
        results.append(result)
        
        # 保存結果 (附加模式)
        results_df = pd.DataFrame([result])
        csv_path = os.path.join(OUTPUT_DIR, 'training_records.csv')
        if os.path.exists(csv_path):
            results_df.to_csv(csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            results_df.to_csv(csv_path, mode='w', header=True, index=False, encoding='utf-8-sig')
    
    # 最終結果
    print("\n" + "="*60)
    print("所有實驗完成!")
    print("="*60)
    
    all_results_df = pd.DataFrame(results)
    print("\n結果摘要:")
    print(all_results_df[['模型類型', '測試準確率', '測試F1分數', '訓練時間(分鐘)']].to_string(index=False))
    
    # 找出最佳模型
    best_idx = all_results_df['測試準確率'].astype(float).idxmax()
    best_result = all_results_df.iloc[best_idx]
    
    print("\n" + "="*60)
    print("🏆 最佳模型:")
    print("="*60)
    print(f"模型類型: {best_result['模型類型']}")
    print(f"測試準確率: {best_result['測試準確率']}")
    print(f"測試 F1: {best_result['測試F1分數']}")
    print("="*60)


if __name__ == '__main__':
    run_all_experiments()