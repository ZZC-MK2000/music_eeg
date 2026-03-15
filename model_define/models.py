import torch.nn as nn


class EEG1DCNN(nn.Module):
    def __init__(self, _n_channels: int, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class EEGResMLPNet(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, hidden_dim: int = 256, depth: int = 4, dropout: float = 0.25):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(*[ResidualMLPBlock(hidden_dim, dropout=dropout) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w


class MultiScaleResBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        self.b3 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.b7 = nn.Conv1d(channels, channels, kernel_size=7, padding=3 * dilation, dilation=dilation)
        self.b15 = nn.Conv1d(channels, channels, kernel_size=15, padding=7 * dilation, dilation=dilation)
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.SiLU()
        self.se = SEBlock(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.b3(x) + self.b7(x) + self.b15(x)
        y = self.norm(y)
        y = self.act(y)
        y = self.se(y)
        y = self.drop(y)
        return x + y


class EEGMSResNet1D(nn.Module):
    def __init__(self, _input_len: int, n_classes: int, width: int = 64, depth: int = 4, dropout: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, width, kernel_size=9, padding=4),
            nn.BatchNorm1d(width),
            nn.SiLU(),
        )
        dilations = [1, 2, 4, 8][: max(1, depth)]
        self.blocks = nn.Sequential(*[MultiScaleResBlock(width, dilation=d, dropout=dropout) for d in dilations])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, n_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)
