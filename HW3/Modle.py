import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class RnnClassfliter(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RnnClassfliter, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 設定初始隱藏狀態和細胞狀態
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向傳播 RNN
        out, _ = self.rnn(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # 取最後一個時間步的輸出
        out = out[:, -1, :]

        # 通過全連接層
        out = self.fc(out)
        return out

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 增加3層全連接層
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 設定初始隱藏狀態和細胞狀態
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向傳播 LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # 取最後一個時間步的輸出
        out = out[:, -1, :]
        # 增加3層全連接層
        out = nn.functional.relu(self.fc1(out))
        out = nn.functional.relu(self.fc2(out))
        out = nn.functional.relu(self.fc3(out))
        # 通過全連接層
        out = self.fc(out)
        return out

# ===== 特徵提取器（使用預訓練模型） =====
class FeatureExtractor(nn.Module):
    """
    使用預訓練的 Transformer 模型提取文本特徵
    凍結預訓練模型參數，只提取特徵向量
    
    支援三種池化方式:
    - 'cls': 使用 [CLS] token (BERT 標準做法，推薦)
    - 'mean': 平均池化 (對所有有效 token 取平均)
    - 'last': 使用最後一個有效 token
    """
    def __init__(self, model_name, use_auth_token=None, pooling='cls'):
        super(FeatureExtractor, self).__init__()
        
        # 載入預訓練模型
        self.model = AutoModel.from_pretrained(model_name, use_auth_token=use_auth_token)
        
        # 設定池化方式
        self.pooling = pooling
        
        # 設定 pad_token_id (處理多種情況)
        if self.model.config.pad_token_id is None:
            if hasattr(self.model.config, 'eos_token_id') and self.model.config.eos_token_id is not None:
                self.model.config.pad_token_id = self.model.config.eos_token_id
            elif hasattr(self.model.config, 'unk_token_id') and self.model.config.unk_token_id is not None:
                self.model.config.pad_token_id = self.model.config.unk_token_id
            else:
                # 設為 0 (通常是 [PAD] token 的 ID)
                self.model.config.pad_token_id = 0
        
        # 凍結模型參數（不進行微調）
        for param in self.model.parameters():
            param.requires_grad = False
        
    def forward(self, input_ids, attention_mask=None):
        """
        提取文本的特徵向量
        根據 pooling 方式返回不同的特徵表示
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs['last_hidden_state'].to(torch.float32)
        
        if self.pooling == 'cls':
            # 使用 [CLS] token (第一個位置)
            # BERT 等模型在預訓練時優化 [CLS] 作為句子表示
            return hidden_states[:, 0, :]
        
        elif self.pooling == 'mean':
            # 平均池化：對所有有效 token 取平均
            # 這種方法能充分利用所有 token 的信息
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                return sum_embeddings / sum_mask
            else:
                return torch.mean(hidden_states, dim=1)
        
        elif self.pooling == 'last':
            # 使用最後一個有效 token
            if attention_mask is not None:
                sequence_lengths = torch.eq(attention_mask, 0).int().argmax(-1) - 1
                sequence_lengths = torch.clamp(sequence_lengths, min=0)
                return hidden_states[
                    torch.arange(hidden_states.shape[0], device=hidden_states.device), 
                    sequence_lengths
                ]
            else:
                return hidden_states[:, -1, :]
        
        else:
            raise ValueError(f"不支援的池化方式: {self.pooling}。請使用 'cls', 'mean', 或 'last'。")


# ===== 情緒分類模型（整合特徵提取 + 分類器） =====
class SentimentClassifier(nn.Module):
    """
    完整的情緒分類模型
    包含特徵提取器和分類器兩部分
    """
    def __init__(self, model_name, hidden_size=256, num_classes=2, dropout_rate=0.3, use_auth_token=None, tokenizer=None, pooling='cls'):
        super(SentimentClassifier, self).__init__()
        
        # 特徵提取器（凍結參數）
        self.feature_extractor = FeatureExtractor(model_name, use_auth_token, pooling=pooling)
        
        # 如果提供了 tokenizer,調整 embedding 大小
        if tokenizer is not None:
            self.feature_extractor.model.resize_token_embeddings(len(tokenizer))
            # 重新凍結
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        
        # 獲取特徵維度（預訓練模型的 hidden size）
        self.feature_dim = self.feature_extractor.model.config.hidden_size
        
        # 分類器部分（可訓練）
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, input_ids, attention_mask=None):
        """
        前向傳播
        input_ids: tokenized 文本
        attention_mask: 注意力遮罩
        """
        # 提取特徵
        features = self.feature_extractor(input_ids, attention_mask)
        
        # 分類
        logits = self.classifier(features)
        
        return logits
    
    def freeze_feature_extractor(self):
        """凍結特徵提取器參數"""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def unfreeze_feature_extractor(self):
        """解凍特徵提取器參數（用於微調）"""
        for param in self.feature_extractor.parameters():
            param.requires_grad = True


# ===== LSTM 版本的情緒分類模型 =====
class LSTMSentimentClassifier(nn.Module):
    """
    使用 LSTM 的情緒分類模型
    特徵提取 -> LSTM -> 分類
    """
    def __init__(self, model_name, hidden_size=128, num_layers=2, num_classes=2, dropout_rate=0.3, use_auth_token=None, tokenizer=None, pooling='cls'):
        super(LSTMSentimentClassifier, self).__init__()
        
        # 特徵提取器
        self.feature_extractor = FeatureExtractor(model_name, use_auth_token, pooling=pooling)
        
        # 如果提供了 tokenizer,調整 embedding 大小
        if tokenizer is not None:
            self.feature_extractor.model.resize_token_embeddings(len(tokenizer))
            # 重新凍結
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        
        # 獲取特徵維度
        self.feature_dim = self.feature_extractor.model.config.hidden_size
        
        # LSTM 層
        self.lstm = nn.LSTM(self.feature_dim, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        
       # 分類器部分（可訓練）
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )
        
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
    def forward(self, input_ids, attention_mask=None):
        """
        前向傳播
        """
        # 提取特徵 (batch_size, feature_dim)
        features = self.feature_extractor(input_ids, attention_mask)
        
        # 擴展維度以適應 LSTM: (batch_size, seq_len=1, feature_dim)
        features = features.unsqueeze(1)
        
        # 初始化 LSTM 隱藏狀態
        h0 = torch.zeros(self.num_layers, features.size(0), self.hidden_size, device=features.device)
        c0 = torch.zeros(self.num_layers, features.size(0), self.hidden_size, device=features.device)
        
        # LSTM 處理
        lstm_out, _ = self.lstm(features, (h0, c0))
        
        # 取最後一個時間步
        lstm_out = lstm_out[:, -1, :]
        
        # 分類
        logits = self.classifier(lstm_out)
        
        return logits


# ===== GRU 版本的情緒分類模型 =====
class GRUSentimentClassifier(nn.Module):
    """
    使用 GRU 的情緒分類模型
    特徵提取 -> GRU -> 分類
    
    GRU 相比 LSTM:
    - 參數更少,訓練更快
    - 只有重置門和更新門,沒有細胞狀態
    - 在許多任務上性能接近 LSTM
    """
    def __init__(self, model_name, hidden_size=128, num_layers=2, num_classes=2, dropout_rate=0.3, use_auth_token=None, tokenizer=None, pooling='cls'):
        super(GRUSentimentClassifier, self).__init__()
        
        # 特徵提取器
        self.feature_extractor = FeatureExtractor(model_name, use_auth_token, pooling=pooling)
        
        # 如果提供了 tokenizer,調整 embedding 大小
        if tokenizer is not None:
            self.feature_extractor.model.resize_token_embeddings(len(tokenizer))
            # 重新凍結
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        
        # 獲取特徵維度
        self.feature_dim = self.feature_extractor.model.config.hidden_size
        
        # GRU 層
        self.gru = nn.GRU(
            self.feature_dim, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=False  # 可以設為 True 使用雙向 GRU
        )
        
        # 分類器部分（可訓練）
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
    def forward(self, input_ids, attention_mask=None):
        """
        前向傳播
        """
        # 提取特徵 (batch_size, feature_dim)
        features = self.feature_extractor(input_ids, attention_mask)
        
        # 擴展維度以適應 GRU: (batch_size, seq_len=1, feature_dim)
        features = features.unsqueeze(1)
        
        # 初始化 GRU 隱藏狀態
        h0 = torch.zeros(self.num_layers, features.size(0), self.hidden_size, device=features.device)
        
        # GRU 處理
        gru_out, _ = self.gru(features, h0)
        
        # 取最後一個時間步
        gru_out = gru_out[:, -1, :]
        
        # 分類
        logits = self.classifier(gru_out)
        
        return logits


# ===== 雙向 GRU 情緒分類模型 =====
class BiGRUSentimentClassifier(nn.Module):
    """
    使用雙向 GRU 的情緒分類模型
    特徵提取 -> BiGRU -> 注意力機制 -> 分類
    
    雙向 GRU:
    - 同時考慮前向和後向上下文
    - 更好地捕捉序列信息
    - 輸出維度是單向的 2 倍
    """
    def __init__(self, model_name, hidden_size=128, num_layers=2, num_classes=2, dropout_rate=0.3, use_auth_token=None, tokenizer=None, pooling='cls'):
        super(BiGRUSentimentClassifier, self).__init__()
        
        # 特徵提取器
        self.feature_extractor = FeatureExtractor(model_name, use_auth_token, pooling=pooling)
        
        # 如果提供了 tokenizer,調整 embedding 大小
        if tokenizer is not None:
            self.feature_extractor.model.resize_token_embeddings(len(tokenizer))
            # 重新凍結
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        
        # 獲取特徵維度
        self.feature_dim = self.feature_extractor.model.config.hidden_size
        
        # 雙向 GRU 層
        self.bigru = nn.GRU(
            self.feature_dim, 
            hidden_size // 2,  # 因為是雙向,最終輸出會是 hidden_size
            num_layers, 
            batch_first=True, 
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 注意力層
        self.attention = nn.Linear(hidden_size, 1)
        
        # 分類器部分（可訓練）
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
    def forward(self, input_ids, attention_mask=None):
        """
        前向傳播
        """
        # 提取特徵 (batch_size, feature_dim)
        features = self.feature_extractor(input_ids, attention_mask)
        
        # 擴展維度以適應 GRU: (batch_size, seq_len=1, feature_dim)
        features = features.unsqueeze(1)
        
        # 初始化 BiGRU 隱藏狀態 (num_layers * 2 because bidirectional)
        h0 = torch.zeros(self.num_layers * 2, features.size(0), self.hidden_size // 2, device=features.device)
        
        # BiGRU 處理
        gru_out, _ = self.bigru(features, h0)  # (batch, seq_len, hidden_size)
        
        # 注意力機制
        attention_scores = self.attention(gru_out)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # 加權平均
        weighted_output = (gru_out * attention_weights).sum(dim=1)  # (batch, hidden_size)
        
        # 分類
        logits = self.classifier(weighted_output)
        
        return logits
















