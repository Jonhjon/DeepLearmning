import torch.nn as nn
import torch.nn.functional as F
class MLP(nn.Module):
    def __init__(self,intput_dim=42, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(intput_dim, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.act1 = nn.ReLU()

        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.act2 = nn.ReLU()

        self.fc3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.3)
        self.act3 = nn.ReLU()

        self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.act3(x)
        x = self.dropout3(x)
        x= self.fc4(x)
        return x

class MLP_deep(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.net(x)