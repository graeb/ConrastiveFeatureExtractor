import lightning as L


class TwoStageTrainingCallback(L.Callback):
    def __init__(self, freeze_epochs, fine_tune_epochs, lr_stage_1, lr_stage_2):
        super().__init__()
        self.freeze_epochs = freeze_epochs
        self.fine_tune_epochs = fine_tune_epochs
        self.lr_stage_1 = lr_stage_1  # Learning rate for stage 1
        self.lr_stage_2 = lr_stage_2  # Learning rate for stage 2
        self.current_stage = "freeze"

    def on_train_epoch_end(self, trainer, pl_module):
        # Transition to fine-tuning stage
        if (
            trainer.current_epoch == self.freeze_epochs - 1
            and self.current_stage == "freeze"
        ):
            self.current_stage = "fine_tune"
            pl_module.unfreeze_backbone()  # Unfreeze the backbone
            self._update_learning_rate(trainer, self.lr_stage_2)  # Update learning rate
            print(f"Transitioning to fine-tune stage with LR={self.lr_stage_2}")

    def on_fit_start(self, trainer, pl_module):
        # Initial setup: Freeze the backbone and set initial LR
        if self.current_stage == "freeze":
            pl_module.freeze_backbone()  # Freeze the backbone
            self._update_learning_rate(
                trainer, self.lr_stage_1
            )  # Set initial learning rate
            print(f"Starting frozen backbone stage with LR={self.lr_stage_1}")

    def _update_learning_rate(self, trainer, new_lr):
        # Update learning rate for all optimizers
        for optimizer in trainer.optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr
