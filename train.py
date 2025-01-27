import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_model import PDNModel


# Custom Lightning Module for contrastive learning with triplet loss
class ContrastiveLearningModule(L.LightningModule):
    def __init__(self, backbone, embedding_dim=128, lr=1e-3, margin=1.0):
        super().__init__()
        self.backbone = backbone

        # MLP adapter: Projects backbone output to the embedding space
        self.mlp = nn.Sequential(
            nn.Linear(384, 256),  # Adjust input size based on backbone output
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
        )

        self.lr = lr
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)

    def forward(self, x):
        """Forward pass through the backbone and MLP adapter."""
        features = self.backbone(x)
        backbone_output = features["layer4"]  # Assuming `layer4` is the output
        backbone_output = backbone_output.view(backbone_output.size(0), -1)  # Flatten
        embedding = self.mlp(backbone_output)
        return F.normalize(embedding, p=2, dim=1)  # Normalize embeddings to unit sphere

    def training_step(self, batch, batch_idx):
        """Training step for triplet loss."""
        anchor, positive, negative = batch
        anchor_emb = self(anchor)
        positive_emb = self(positive)
        negative_emb = self(negative)

        loss = self.triplet_loss(anchor_emb, positive_emb, negative_emb)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for triplet loss."""
        anchor, positive, negative = batch
        anchor_emb = self(anchor)
        positive_emb = self(positive)
        negative_emb = self(negative)

        loss = self.triplet_loss(anchor_emb, positive_emb, negative_emb)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        """Configure optimizers."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


# Custom Dataset for Triplet Sampling
class TripletDataset(Dataset):
    def __init__(self, data, transform=None):
        """
        Args:
            data (list of tuples): Each tuple contains (anchor, positive, negative) samples.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor, positive, negative = self.data[idx]
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        return anchor, positive, negative


# Example training script
if __name__ == "__main__":
    # Instantiate your custom backbone and load weights
    backbone = PDNModel(out_channels=384, padding=False)
    # Load pre-trained weights if available
    # backbone.load_state_dict(torch.load("path_to_weights.pth"))

    # Define a LightningModule
    model = ContrastiveLearningModule(backbone=backbone)

    # Example DataLoader with dummy data
    dummy_data = [
        (torch.randn(3, 64, 64), torch.randn(3, 64, 64), torch.randn(3, 64, 64))
        for _ in range(100)
    ]
    train_dataset = TripletDataset(dummy_data)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Validation DataLoader
    val_dataset = TripletDataset(dummy_data)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Train the model
    trainer = L.Trainer(max_epochs=10, gpus=1 if torch.cuda.is_available() else 0)
    trainer.fit(model, train_loader, val_loader)
