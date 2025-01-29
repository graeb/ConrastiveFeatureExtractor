import lightning as L
import torch


class TwoStageTrainingCallback(L.Callback):
    def __init__(
        self,
        freeze_epochs,
        fine_tune_epochs,
        lr_stage_1,
        lr_stage_2,
        scheduler_step_epochs=5,
    ):
        super().__init__()
        self.freeze_epochs = freeze_epochs
        self.fine_tune_epochs = fine_tune_epochs
        self.lr_stage_1 = lr_stage_1  # Learning rate for stage 1
        self.lr_stage_2 = lr_stage_2  # Learning rate for stage 2
        self.scheduler_step_epochs = scheduler_step_epochs
        self.current_stage = "freeze"

    def on_train_epoch_end(self, trainer, pl_module):
        """Handles transition between training stages & LR scheduling."""
        if (
            trainer.current_epoch == self.freeze_epochs - 1
            and self.current_stage == "freeze"
        ):
            self.current_stage = "fine_tune"
            pl_module.unfreeze_backbone()  # Unfreeze the backbone
            self._update_learning_rate(
                trainer, self.lr_stage_2
            )  # Adjust LR for fine-tuning
            print(f"Transitioning to fine-tune stage with LR={self.lr_stage_2}")

        # Step the scheduler every 'scheduler_step_epochs'
        if trainer.current_epoch % self.scheduler_step_epochs == 0:
            self._step_lr_scheduler(trainer)

    def on_fit_start(self, trainer, pl_module):
        """Initial setup: Freeze the backbone and set the starting LR."""
        if self.current_stage == "freeze":
            pl_module.freeze_backbone()
            self._update_learning_rate(trainer, self.lr_stage_1)
            print(f"Starting frozen backbone stage with LR={self.lr_stage_1}")

    def _update_learning_rate(self, trainer, new_lr):
        """Updates the learning rate for all optimizers."""
        for optimizer in trainer.optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr

    def _step_lr_scheduler(self, trainer):
        """Steps the LR scheduler correctly, ensuring required metrics are passed."""
        if trainer.lr_scheduler_configs:
            for scheduler_config in trainer.lr_scheduler_configs:
                scheduler = scheduler_config.scheduler
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # Pass the validation loss if available
                    if "val_loss" in trainer.callback_metrics:
                        scheduler.step(trainer.callback_metrics["val_loss"])
                        print(
                            f"ReduceLROnPlateau stepped with val_loss={trainer.callback_metrics['val_loss']}"
                        )
                    else:
                        print(
                            "Warning: ReduceLROnPlateau requires a metric but none was found."
                        )
                else:
                    scheduler.step()  # Regular schedulers don't need a metric
                    print(f"LR Scheduler stepped at epoch {trainer.current_epoch}")
