import lightning as L
import argparse
from lightning.pytorch.loggers import WandbLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import TripletVisA
from callbacks import TwoStageTrainingCallback
from utils import load_pdn, load_visa_dataset

from torch_model import PDNModel


class ContrastiveLearningModule(L.LightningModule):
    def __init__(
        self,
        backbone,
        lr=1e-3,
        margin=1.0,
        normalize_projection=True,
    ):
        super().__init__()
        self.normalize_projection = normalize_projection
        # self.freeze_backbone = freeze_backbone
        self.backbone = backbone

        # feature adapter: Projects backbone output to the embedding space - watchout for vanishing gradient
        self.projection_head = nn.Linear(384 * 56 * 56, 384)
        self.lr = lr
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)

        # Freeze the backbone initially
        # if freeze_backbone:
        #     for param in self.backbone.parameters():
        #         param.requires_grad = False

    def forward(self, x):
        """Forward pass through the backbone and MLP adapter."""
        features = self.backbone(x)
        backbone_output = features["layer4"]
        print("Backbone feature shape", backbone_output.shape)
        backbone_output = backbone_output.view(backbone_output.size(0), -1)
        embedding = self.projection_head(backbone_output)
        if self.normalize_projection:
            return F.normalize(
                embedding, p=2, dim=1
            )  # Normalize embeddings to unit sphere
        else:
            return embedding

    def training_step(self, batch, batch_idx):
        """Training step for triplet loss."""
        anchor, positive, negative = batch
        anchor_emb = self(anchor)
        positive_emb = self(positive)
        negative_emb = self(negative)

        loss = self.triplet_loss(anchor_emb, positive_emb, negative_emb)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def validation_step(self, batch, batch_idx):
        """Validation step for triplet loss."""
        anchor, positive, negative = batch
        anchor_emb = self(anchor)
        positive_emb = self(positive)
        negative_emb = self(negative)

        pos_dist = F.pairwise_distance(anchor_emb, positive_emb, p=2)  # L2 distance
        neg_dist = F.pairwise_distance(anchor_emb, negative_emb, p=2)

        self.log("neg_dist", neg_dist, on_step=False, on_epoch=True)
        self.log("pos_div", pos_dist, on_step=False, on_epoch=True)

        loss = self.triplet_loss(anchor_emb, positive_emb, negative_emb)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

        accuracy = (pos_dist < neg_dist).float().mean()
        self.log("val_accuracy", accuracy, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    # def configure_optimizers(self):
    #     """Configure optimizers with a different learning rate for backbone and projection head."""
    #     params = [
    #         {
    #             "params": self.projection_head.parameters(),
    #             "lr": self.lr,
    #         },  # High LR for MLP
    #     ]
    #     if not self.freeze_backbone:
    #         params.append(
    #             {"params": self.backbone.parameters(), "lr": self.lr * 0.1}
    #         )  # Lower LR for backbone
    #     optimizer = torch.optim.Adam(params)
    #     return optimizer
    #
    # def unfreeze_backbone(self):
    #     """Unfreeze the backbone for fine-tuning."""
    #     for param in self.backbone.parameters():
    #         param.requires_grad = True
    #     self.freeze_backbone = False


# Example training script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Fine tuning backbone for effective feature extraction"
    )
    parser.add_argument("-n", "--name", type=str, required=True)
    parser.add_argument("-p", "--path", type=str, required=True)
    parser.add_argument("-b", "--batch-size", type=int, required=True)
    args = parser.parse_args()

    logger = WandbLogger(name=args.name, project="ContrastiveBackboneAD")
    callback = TwoStageTrainingCallback(
        freeze_epochs=100, fine_tune_epochs=1000, lr_stage_1=1e-3, lr_stage_2=1e-4
    )
    # Instantiate your custom backbone and load weights
    backbone = load_pdn()

    # Define a LightningModule
    model = ContrastiveLearningModule(
        backbone=backbone, lr=1e-3, normalize_projection=True
    )

    # Load data
    load_visa_dataset(args.path)
    train_dataset = TripletVisA(path=args.path, split="train", val_split=0.2)
    val_dataset = TripletVisA(path=args.path, split="val", val_split=0.2)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Train the model
    trainer = L.Trainer(
        callbacks=callback,
        max_epochs=1100,
        logger=logger,
        devices="auto",
    )
    trainer.fit(model, train_loader, val_loader)

    # Second stage unfreeze backbone weights
    # model.unfreeze_backbone()
    # model.lr = 1e-4
    # trainer.fit(model, train_loader, val_loader)
