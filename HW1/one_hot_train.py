import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from Modle import MLP
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Num GPUs Available:", torch.cuda.device_count())

path = r'C:\Users\H514 #4856\Desktop\deep learning 114206103\HW1\archive/dataset.csv'

binary_cols =['Gender','Daytime_evening_attendance','Displaced','Educational special needs','Debtor'
              ,'Tuition fees up to date','Scholarship holder','International']

categorical_cols = ['Marital status','Application mode','Course'
                    ,'Previous qualification','Nacionality','Mother qualification'
                    ,'Father qualification','Mother occupation'
                    ,'Father occupation']

numeric_cols =['Application order','Age at enrollment','Curricular units 1st sem (credited)'
               ,'Curricular units 1st sem (enrolled)','Curricular units 1st sem (evaluations)',
               'Curricular units 1st sem (approved)','Curricular units 1st sem (grade)',
               'Curricular units 1st sem (without evaluations)',
               'Curricular units 2nd sem (credited)','Curricular units 2nd sem (enrolled)',
               'Curricular units 2nd sem (evaluations)','Curricular units 2nd sem (approved)',
               'Curricular units 2nd sem (grade)','Curricular units 2nd sem (without evaluations)',
               'Unemployment rate','Inflation rate','GDP']

df = pd.read_csv(path)
print(f"Original DataFrame shape: {df.shape}")

# (1) 對 0/1 類別型做 One-Hot
df[binary_cols] = df[binary_cols].astype(str)
df = pd.get_dummies(df, columns=binary_cols, drop_first=True)

#對一般類別型欄位做 One-Hot
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# (3) 對數值型欄位做 Min-Max 正規化
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
print(f"Transformed DataFrame shape: {df.shape}")

X_df = df.drop('Target', axis=1).apply(pd.to_numeric, errors='coerce').fillna(0)
X = X_df.values.astype(np.float32)
y = df['Target'].values
# print(X.shape)
# print(f"Transformed DataFrame shape: {df.shape}")

le = LabelEncoder()
y = le.fit_transform(y)
# print(y)  # (4209, 57) (4209, 56) (4209,)
# 從訓練的CSV中分出訓練跟測試7:3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# 從訓練資料中再分出訓練跟驗證8:2
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# 轉成 tensor
# X_train = torch.tensor(X_train, dtype=torch.float32)
# y_train = torch.tensor(y_train, dtype=torch.long)
# X_val = torch.tensor(X_val, dtype=torch.float32)
# y_val = torch.tensor(y_val, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# 建立 DataLoader
# train_dataset = TensorDataset(X_train, y_train)
# val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)
# train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 設定參數
# --- 修正後的 K-Fold 訓練流程 ---

# 設定參數
n_splits = 5
num_epochs = 1000
lr= 0.001 # 【註】: lr=1 對 SGD 來說非常高，如果不穩定，請嘗試 0.1 或 0.01
task = 'lr_test'
print(X.shape[1], len(np.unique(y)))

# 【建議】: 驗證和測試時，batch_size 可以設大一點，會跑比較快
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False) # 原本是 64

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
best_models = []

# 【註】: 這兩個 list (best_train_loss, best_val_loss) 
# 不再需要在 K-fold 迴圈開始前宣告
# best_train_loss = [] (刪除)
# best_val_loss = [] (刪除)


for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):

    model = MLP(intput_dim=X.shape[1], num_classes=len(np.unique(y))).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\n--- Fold {fold}/{n_splits} ---")
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]

    X_tr = torch.tensor(X_tr, dtype=torch.float32)
    y_tr = torch.tensor(y_tr, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    # 建立 DataLoader
    train_dataset = TensorDataset(X_tr, y_tr)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    # 【建議】: 驗證的 batch_size 也可以調大
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False) # 原本是 64

    # 這些 list 用於儲存「這一個 fold」的歷史
    train_loss = []
    train_acc = []
    val_loss_list = []

    best_val_acc_fold = -1.0
    best_state_fold = None
    best_epoch = -1
    
    # 【新增】: 用於儲存「這一個 fold」的「最佳」loss 歷史
    fold_best_train_loss_history = []
    fold_best_val_loss_history = []

    # 訓練模型
    for epoch in range(num_epochs):
        epoch_train_loss = 0.0
        epoch_train_corrects = 0
        epoch_train_samples = 0
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() * batch_X.size(0)
            epoch_train_corrects += (outputs.argmax(dim=1) == batch_y).sum().item()
            epoch_train_samples += batch_X.size(0)
            
        avg_epoch_train_loss = epoch_train_loss / epoch_train_samples
        avg_epoch_train_acc = epoch_train_corrects / epoch_train_samples
        
        train_loss.append(avg_epoch_train_loss) 
        train_acc.append(avg_epoch_train_acc)
        
        # 驗證模型
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())

        val_loss /= len(val_loader)
        val_loss_list.append(val_loss)
        val_acc = np.mean(np.array(val_preds) == np.array(val_labels))

        if val_acc > best_val_acc_fold:
            best_val_acc_fold = val_acc
            best_state_fold = {k:v.cpu() for k,v in model.state_dict().items()}
            best_epoch = epoch + 1
            
            # 【修正】: 關鍵錯誤 (2) - 使用 .copy() 來「快照」當前最佳 epoch 的歷史
            fold_best_train_loss_history = train_loss.copy()
            fold_best_val_loss_history = val_loss_list.copy()

        # 每 10 個 epoch 印一次結果 (可選)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, "
                  f"train_loss : {avg_epoch_train_loss:.4f}, "
                  f"train_acc : {avg_epoch_train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val acc: {val_acc:.4f}")
    
    # --- Epoch 迴圈結束 ---

    # 【修正】: 關鍵錯誤 (1) - `best_models.append` 必須在 Epoch 迴圈「外」，Fold 迴圈「內」
    if best_state_fold is not None:
        # 把這個 fold 的最佳成績、模型狀態、fold編號、最佳loss歷史 全都存起來
        best_models.append((best_val_acc_fold, best_state_fold, fold, fold_best_train_loss_history, fold_best_val_loss_history))

# --- K-Fold 迴圈結束 ---


if len(best_models) == 0:
    raise RuntimeError("No trained models saved from folds.")

# 【修正】: 解構元組 (tuple) 時，取出我們為繪圖儲存的 loss 列表
best_val_acc, best_state, best_fold, final_plot_train_loss, final_plot_val_loss = max(best_models, key=lambda x: x[0])
print(f"\nSelected best fold: {best_fold} with val acc: {best_val_acc:.4f}")

base = os.path.join(r'C:\Users\H514 #4856\Desktop\deep learning 114206103\\HW1', task)

if not os.path.exists(base):
    os.makedirs(base)

# 【修正】: 現在畫圖時，使用的是剛剛從 best_models 解構出來的「最終」列表
plt.figure(figsize=(10, 6))
plt.plot(final_plot_train_loss, label='Train Loss')
plt.plot(final_plot_val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f'Training and Validation Loss (Best Fold: {best_fold})')
plt.legend()
plt.grid()
plt.savefig(os.path.join(base,f"loss_curve_best_fold_{best_fold}_lr_0.001_Adam.jpg"))

# 在測試集上評估模型 (你這一段的邏輯已經是正確的，保持不變)
best_model = MLP(intput_dim=X.shape[1], num_classes=len(np.unique(y))).to(device)
best_model.load_state_dict(best_state)
best_model.eval()
test_preds = []
test_labels = []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = best_model(batch_X)
        test_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        test_labels.extend(batch_y.cpu().numpy())

test_precision = precision_score(y_test.numpy(), test_preds, average='weighted')
print(classification_report(y_test.numpy(), test_preds))
test_recall = recall_score(y_test.numpy(), test_preds, average='weighted')
test_f1 = f1_score(y_test.numpy(), test_preds, average='weighted')
test_acc = np.mean(np.array(test_preds) == np.array(test_labels))
print(f"(best model from fold {best_fold})\n"
      f"Test Accuracy : {test_acc:.4f},\n"
      f"Test Precision : {test_precision:.4f},\n"
      f"Test Recall: {test_recall:.4f},\n"
      f"Test F1 Score: {test_f1:.4f}\n")