import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import argparse
from lightning.pytorch.loggers import WandbLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import TripletVisA
from projection_heads import (
    ProjectionHeadBN,
    ProjectionHeadCW,
    ProjectionHeadMaxMargin,
    ProjectionHeadResMLP,
    ProjectionHeadSSN,
)
from callbacks import TwoStageTrainingCallback
import projection_heads
from utils import load_pdn, load_visa_dataset
from itertools import chain


class ContrastiveLearningModule(L.LightningModule):
    def __init__(
        self,
        projection_head,
        backbone,
        projection_type="rest",
        lr=1e-3,
        margin=1.0,
    ):
        super().__init__()
        # self.freeze_backbone = freeze_backbone
        self.backbone = backbone

        # feature adapter: Projects backbone output to the embedding space - watchout for vanishing gradient
        self.projection_head = projection_head
        self.projection_type = projection_type
        self.lr = lr
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)

    def forward(self, x):
        """Forward pass through the backbone and MLP adapter."""
        features = self.backbone(x)
        backbone_output = features["layer4"]
        if self.projection_type != "SSN":
            backbone_output = backbone_output.view(backbone_output.size(0), -1)
        embedding = self.projection_head(backbone_output)
        return F.normalize(embedding, p=2, dim=1)

    def training_step(self, batch, batch_idx):
        """Training step for triplet loss."""
        anchor, positive, negative = batch
        anchor_emb = self(anchor)
        positive_emb = self(positive)
        negative_emb = self(negative)

        # Check gradients for exploding/vanishing
        grad_norms = []
        for param in chain(
            self.backbone.parameters(), self.projection_head.parameters()
        ):
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())

        # Log max and min gradient norms
        if grad_norms:
            max_grad = max(grad_norms)
            min_grad = min(grad_norms)
            avg_grad = sum(grad_norms) / len(grad_norms)

            self.log("grad_norm/max", max_grad)
            self.log("grad_norm/min", min_grad)
            self.log("grad_norm/avg", avg_grad)

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

        self.log("mean_neg_dist", neg_dist.mean(), on_step=False, on_epoch=True)
        self.log("mean_pos_div", pos_dist.mean(), on_step=False, on_epoch=True)

        self.log("var_neg_dist", neg_dist.var(), on_step=False, on_epoch=True)
        self.log("var_pos_div", pos_dist.var(), on_step=False, on_epoch=True)

        loss = self.triplet_loss(anchor_emb, positive_emb, negative_emb)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

        accuracy = (pos_dist < neg_dist).float().mean()
        self.log("val_accuracy", accuracy, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
            },
        }

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
    # TODO: We could potentially load masks from negative pairs and somehow
    # incentivise loss function to focus on that part something akin to focal
    # loss would need to be added to triplet oss

    parser = argparse.ArgumentParser(
        "Fine tuning backbone for effective feature extraction"
    )
    parser.add_argument("-n", "--name", type=str, required=True)
    parser.add_argument("-p", "--path", type=str, required=True)
    parser.add_argument("-b", "--batch-size", type=int, required=True)
    args = parser.parse_args()

    # Fails! too much data     "Attn": ProjectionHeadAttention(),
    projection_heads = {
        "BN": ProjectionHeadBN(),
        "CW": ProjectionHeadCW(),
        "MaxM": ProjectionHeadMaxMargin(),
        "ResMLP": ProjectionHeadResMLP(),
        "SSN": ProjectionHeadSSN(),
    }

    # Instantiate your custom backbone and load weights
    backbone = load_pdn()

    for suffix, projection_head in projection_heads.items():
        run_name = f"{args.name}-{suffix}"
        logger = WandbLogger(name=run_name, project="ContrastiveBackboneAD")

        checkpoint = ModelCheckpoint(
            "./results", monitor="train_loss", every_n_train_steps=30
        )
        scheduling = TwoStageTrainingCallback(
            freeze_epochs=10, fine_tune_epochs=100, lr_stage_1=1e-3, lr_stage_2=1e-4
        )

        # Define a LightningModule
        model = ContrastiveLearningModule(
            projection_head, projection_type=suffix, backbone=backbone
        )

        # Load data
        load_visa_dataset(args.path)
        train_dataset = TripletVisA(path=args.path, split="train", val_split=0.2)
        val_dataset = TripletVisA(path=args.path, split="val", val_split=0.2)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8
        )

        # Train the model
        trainer = L.Trainer(
            callbacks=[checkpoint, scheduling],
            max_epochs=110,
            logger=logger,
            devices="auto",
            # gradient_clip_val=0.5,
        )
        trainer.fit(model, train_loader, val_loader)
        trainer.save_checkpoint(run_name)

        # Second stage unfreeze backbone weights
        # model.unfreeze_backbone()
        # model.lr = 1e-4
        # trainer.fit(model, train_loader, val_loader)
