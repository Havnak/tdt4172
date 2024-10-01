from torch.optim import lr_scheduler, Optimizer
import math


class WarmupCosineDecay(lr_scheduler.LRScheduler):
    def __init__(self, optimizer: Optimizer, warmup_steps, total_steps, min_lr=0, last_epoch=-1, verbose="deprecated"):
        """
        Args:
            optimizer: Optimizer to apply the learning rate schedule to.
            warmup_steps: Number of steps for warmup phase.
            total_steps: Total number of training steps (warmup + decay).
            min_lr: Minimum learning rate at the end of the cosine decay.
            last_epoch: The index of the last epoch (default: -1).
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.optimizer = optimizer
        super().__init__(optimizer, last_epoch, verbose)
        
    def get_lr(self):
        step = self.last_epoch + 1
        if step < self.warmup_steps:
            warmup_factor = step/self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        else:
            decay_step = step - self.warmup_steps
            total_decay_steps = self.total_steps - self.warmup_steps

            cosine_decay_factor = 0.5 * (1 + math.cos(math.pi*(decay_step/total_decay_steps)))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_decay_factor
                for base_lr in self.base_lrs
            ]