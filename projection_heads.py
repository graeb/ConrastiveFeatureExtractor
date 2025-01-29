from torch import nn
from torch.nn import functional as F  # noqa: N812


class ProjectionHeadMaxMargin(nn.Module):
    def __init__(
        self, input_dim=384 * 56 * 56, hidden_dim=512, output_dim=128, margin=0.2
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.margin = margin

    def forward(self, x):
        x = self.projection(x)
        x = F.normalize(x, p=2, dim=1)  # Normalize embeddings
        return x * (1 + self.margin)  # Apply margin scaling


class ProjectionHeadAttention(nn.Module):
    def __init__(self, input_dim=384 * 56 * 56, output_dim=128, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, x):
        x = x.unsqueeze(0)  # Add sequence dimension for attention
        x, _ = self.attn(x, x, x)  # Self-attention
        x = x.squeeze(0)  # Remove sequence dimension
        x = self.projection(x)
        return F.normalize(x, p=2, dim=1)


class ProjectionHeadCW(nn.Module):
    def __init__(self, input_dim=384 * 56 * 56, hidden_dim=512, output_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.eps = 1e-5

    def forward(self, x):
        x = self.projection(x)  # Normal forward
        x_mean = x.mean(dim=0, keepdim=True)  # Mean per batch
        x_std = x.std(dim=0, keepdim=True) + self.eps  # Std per batch
        x = (x - x_mean) / x_std  # Normalize per dimension
        return F.normalize(x, p=2, dim=1)


class ProjectionHeadResMLP(nn.Module):
    def __init__(self, input_dim=384 * 56 * 56, hidden_dim=512, output_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.shortcut = nn.Linear(input_dim, output_dim)  # Skip connection

    def forward(self, x):
        identity = self.shortcut(x)  # Skip connection
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x)) + identity  # Residual connection
        return F.normalize(x, p=2, dim=1)  # Normalize embeddings


class ProjectionHeadSSN(nn.Module):
    def __init__(self, channel_dim: int = 384) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels=channel_dim,
            out_channels=channel_dim,
            kernel_size=1,
            stride=1,
        )

    def forward(self, x):
        return self.projection(x)


class ProjectionHeadBN(nn.Module):
    def __init__(self, input_dim=384 * 56 * 56, hidden_dim=512, output_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),  # Normalize embeddings in feature space
        )

    def forward(self, x):
        return F.normalize(self.projection(x), p=2, dim=1)
