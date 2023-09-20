import torch
import torch.nn as nn
import torch.nn.functional as F

class squeeze_excite(nn.Module):
    def __init__(self, mode, channels, ratio):
        super(squeeze_excite, self).__init__()
        self.avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.max_pooling = nn.AdaptiveMaxPool1d(1)
        if mode == "max":
            self.global_pooling = self.max_pooling
        elif mode == "avg":
            self.global_pooling = self.avg_pooling
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=channels, out_features=channels // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=channels // ratio, out_features=channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        batch_size, channel_size, _= x.shape
        v = self.global_pooling(x).view(batch_size, channel_size)
        v = self.fc_layers(v).view(batch_size, channel_size, 1)
        v = self.sigmoid(v)
        return x * v



class Model(nn.Module):

    def __init__(self,squeeze=16):
        super(Model, self).__init__()
        k1, k2, k3, k4, k5 = 9, 7, 5, 3, 3
        self.squeeze=squeeze
        
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=32, kernel_size=k1, padding=int((k1 - 1) / 2))
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=k2, padding=int((k2 - 1) / 2))
        self.pool2 = nn.MaxPool1d(2)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=k3, padding=int((k3 - 1) / 2))
        self.pool3 = nn.MaxPool1d(2)
        self.bn3 = nn.BatchNorm1d(128)

        self.conv4 = nn.Conv1d(128, 256,  kernel_size=k4, padding=int((k4 - 1) / 2))
        self.bn4 = nn.BatchNorm1d(256)

        self.conv5 = nn.Conv1d(256, 256, kernel_size=k5, padding=int((k5 - 1) / 2))
        self.bn5 = nn.BatchNorm1d(256)
        if self.squeeze:
            self.sqex = squeeze_excite("avg", 256,16 )
        self.dropout = nn.Dropout(p=0.3)
        self.linear1 = nn.Linear(256 * 500, 48)
        self.linear2 = nn.Linear(48, 16)
        self.linear3 = nn.Linear(16, 2)

    def forward(self, x):
        x = x.reshape(x.shape[0], 6, -1)

        x = self.bn1(self.conv1(x))
        x = self.pool(F.relu(x))
        x = self.bn2(self.conv2(x))
        x = self.pool2(F.relu(x))
        x = self.bn3(self.conv3(x))
        x = self.pool3(F.relu(x))
        x = self.bn4(self.conv4(x))
        x = F.relu(x)
        x = self.bn5(self.conv5(x))
        x = F.relu(x)
        if self.squeeze:
            x = self.sqex(x)
        x = x.view(x.shape[0], -1)
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        x = torch.relu(self.linear2(x))
        #x = self.linear2(x)
        x = self.linear3(x)
        return x
'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model().to(device)
summary(model,input_size=(2000,6))
'''
