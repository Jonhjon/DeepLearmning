import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
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
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Num GPUs Available:", torch.cuda.device_count())

# path = r'C:\Users\H514 #4856\Desktop\deep learning 114206103\HW1\high_correlation_features.csv'
#訓練集
train_path = r'C:\Users\H514 #4856\Desktop\deep learning 114206103\HW1\train_set.csv'
train_df = pd.read_csv(train_path)
train_X = train_df.drop('Target', axis=1).values
train_y = train_df['Target'].values

#測試集
test_path = r'C:\Users\H514 #4856\Desktop\deep learning 114206103\HW1\test_set.csv'
test_df = pd.read_csv(test_path)
X_test = test_df.drop('Target', axis=1).values
y_test = test_df['Target'].values


le = LabelEncoder()
train_y = le.fit_transform(train_y)
y_test = le.transform(y_test)
# 從訓練的CSV中分出訓練跟驗證8:2
X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.3, random_state=42, stratify=train_y)


# 轉成 tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)


# 建立 DataLoader
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 建立模型
# model = MLP(input_dim=train_X.shape[1], num_classes=len(np.unique(train_y))).to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

n_splits = 5
num_epochs = 1000
lr= 0.001

# 使用原始 train_X, train_y 做 5-fold CV
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
fold_metrics = []

train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []

# for epoch in range(num_epochs):
#     model.train()
#     running_loss, correct, total = 0.0, 0, 0
#     for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
#         xb, yb = xb.to(device), yb.to(device)
#         optimizer.zero_grad()
#         outputs = model(xb)
#         loss = criterion(outputs, yb)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item() * xb.size(0)
#         _, predicted = torch.max(outputs, 1)
#         total += yb.size(0)
#         correct += (predicted == yb).sum().item()
#     acc = correct / total
#     train_losses.append(running_loss / total)
#     train_accuracies.append(acc)

#     # 驗證
#     model.eval()
#     val_loss, val_correct, val_total = 0.0, 0, 0
#     with torch.no_grad():
#         for xb, yb in val_loader:
#             xb, yb = xb.to(device), yb.to(device)
#             outputs = model(xb)
#             loss = criterion(outputs, yb)
#             val_loss += loss.item() * xb.size(0)
#             _, predicted = torch.max(outputs, 1)
#             val_total += yb.size(0)
#             val_correct += (predicted == yb).sum().item()
#     val_losses.append(val_loss / val_total)
#     val_accuracies.append(val_correct / val_total)

#     print(f"Epoch {epoch+1}/{num_epochs}, "
#           f"Train Loss: {running_loss/total:.4f}, Train Acc: {acc:.4f}, "
#           f"Val Loss: {val_loss/val_total:.4f}, Val Acc: {val_correct/val_total:.4f}")

for fold, (tr_idx, val_idx) in enumerate(skf.split(train_X, train_y), 1):
    print(f"\n--- Fold {fold}/{n_splits} ---")
    X_tr, X_val = train_X[tr_idx], train_X[val_idx]
    y_tr, y_val = train_y[tr_idx], train_y[val_idx]
    # DataLoader
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    train_loader_f = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=256, shuffle=True)
    val_loader_f = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=256, shuffle=False)

    # new model for fold
    model = MLP(input_dim=train_X.shape[1], num_classes=len(np.unique(train_y))).to(device)
    optimizer_f = optim.Adam(model.parameters(), lr=lr)
    criterion_f = nn.CrossEntropyLoss()

    train_losses_f, train_accs_f = [], []
    val_losses_f, val_accs_f = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for xb, yb in tqdm(train_loader_f, desc=f"Fold{fold} Epoch {epoch+1}/{num_epochs}", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            optimizer_f.zero_grad()
            outputs = model(xb)
            loss = criterion_f(outputs, yb)
            loss.backward()
            optimizer_f.step()
            running_loss += loss.item() * xb.size(0)
            _, pred = torch.max(outputs, 1)
            total += yb.size(0)
            correct += (pred == yb).sum().item()
        train_losses_f.append(running_loss / total)
        train_accs_f.append(correct / total)

        # validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader_f:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                loss = criterion_f(outputs, yb)
                val_loss += loss.item() * xb.size(0)
                _, pred = torch.max(outputs, 1)
                val_total += yb.size(0)
                val_correct += (pred == yb).sum().item()
        val_losses_f.append(val_loss / val_total)
        val_accs_f.append(val_correct / val_total)

    # save fold model and plots
    # ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    # fold_dir = os.path.join('results', f'fold_{fold}_{ts}')
    # os.makedirs(fold_dir, exist_ok=True)
    # model_path = os.path.join(fold_dir, f'model_fold{fold}.pt')
    # torch.save(model.state_dict(), model_path)

    # plt.figure()
    # plt.plot(train_losses_f, label='Train Loss')
    # plt.plot(val_losses_f, label='Val Loss')
    # plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title(f'Fold {fold} Loss')
    # plt.savefig(os.path.join(fold_dir, f'loss_fold{fold}_{ts}.png'), dpi=300, bbox_inches='tight')
    # plt.close()

    # plt.figure()
    # plt.plot(train_accs_f, label='Train Acc')
    # plt.plot(val_accs_f, label='Val Acc')
    # plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.title(f'Fold {fold} Acc')
    # plt.savefig(os.path.join(fold_dir, f'acc_fold{fold}_{ts}.png'), dpi=300, bbox_inches='tight')
    # plt.close()

    # fold_metrics.append({'fold': fold, 'val_loss': val_losses_f[-1], 'val_acc': val_accs_f[-1], 'model_path': model_path})
    print(f"Fold {fold} done. Val Loss: {val_losses_f[-1]:.4f}, Val Acc: {val_accs_f[-1]:.4f}")

# 摘要五折結果
val_losses = [m['val_loss'] for m in fold_metrics]
val_accs = [m['val_acc'] for m in fold_metrics]
print("\n=== 5-fold summary ===")
print(f"Val Loss mean: {np.mean(val_losses):.4f}, std: {np.std(val_losses):.4f}")
print(f"Val Acc  mean: {np.mean(val_accs):.4f}, std: {np.std(val_accs):.4f}")

# fold_metrics 已在上面建立，包含 keys: 'fold','val_loss','val_acc','model_path'
best = max(fold_metrics, key=lambda x: x['val_acc'])
best_fold = best['fold']
best_model_path = best['model_path']
print(f"Best fold: {best_fold}, val_acc: {best['val_acc']:.4f}, model_path: {best_model_path}")

# 建立 full train 的 DataLoader（使用原始 train_X, train_y）
X_full = torch.tensor(train_X, dtype=torch.float32)
y_full = torch.tensor(train_y, dtype=torch.long)
full_dataset = TensorDataset(X_full, y_full)
full_loader = DataLoader(full_dataset, batch_size=256, shuffle=True)

model_full = MLP(input_dim=train_X.shape[1], num_classes=len(np.unique(train_y))).to(device)
model_full.load_state_dict(torch.load(best_model_path, map_location=device))

optimizer_full = optim.Adam(model_full.parameters(), lr=lr)
criterion_full = nn.CrossEntropyLoss()

full_train_losses, full_train_accs = [], []

print("\nTraining on full train_set (initialized from best fold) ...")
for epoch in range(epoch):
    model_full.train()
    running_loss, correct, total = 0.0, 0, 0
    for xb, yb in tqdm(full_loader, desc=f"Full Train Epoch {epoch+1}/{epoch}", leave=False):
        xb, yb = xb.to(device), yb.to(device)
        optimizer_full.zero_grad()
        outputs = model_full(xb)
        loss = criterion_full(outputs, yb)
        loss.backward()
        optimizer_full.step()
        running_loss += loss.item() * xb.size(0)
        _, pred = torch.max(outputs, 1)
        total += yb.size(0)
        correct += (pred == yb).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    full_train_losses.append(epoch_loss)
    full_train_accs.append(epoch_acc)
    # if (epoch + 1) % 10 == 0 or epoch == 0:
    #     print(f"Epoch {epoch+1}/{epoch} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

# 儲存 final full-model
os.makedirs('results', exist_ok=True)
final_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
final_model_path = os.path.join('results', f'bestfold{best_fold}_fulltrained_{final_ts}.pt')
torch.save(model_full.state_dict(), final_model_path)
print(f"Saved final full-trained model to {final_model_path}")


# 使用 5 折模型 ensemble 在 test_set 上評估（softmax 平均）
print("\nEvaluating ensemble on test_set (no final full-train)...")
# 載入所有 fold 模型並對 test 做預測平均
model_prob_sum = None
n_models = len(fold_metrics)

# 測試
model_full.eval()
all_preds = []
all_labels = []
correct, total = 0, 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        outputs = model(xb)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy()) 
        all_labels.extend(yb.cpu().numpy())
test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

print(f"Test Accuracy:  {test_acc:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall:    {recall:.4f}")
print(f"Test F1 Score:  {f1:.4f}")

os.makedirs('results', exist_ok=True)
ts = datetime.now().strftime('%Y%m%d_%H%M%S')

# Loss curve
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')
loss_path = os.path.join('results', f'loss_curve_{ts}.png')
plt.savefig(loss_path, dpi=300, bbox_inches='tight')
plt.show()

# Accuracy curve
plt.figure()
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curve')
acc_path = os.path.join('results', f'accuracy_curve_{ts}.png')
plt.savefig(acc_path, dpi=300, bbox_inches='tight')
plt.show()

# 計算混淆矩陣
cm = confusion_matrix(all_labels, all_preds)

# 畫出混淆矩陣
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()