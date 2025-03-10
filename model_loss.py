import torch
import torch.nn as nn

class ReweightingCalculator(nn.Module):
    """
    Computes dynamic loss weights from loss weight calculation layer.
    """
    def __init__(self, num_tasks):
        """
        Args:
            num_tasks (int): Number of multitasking tasks (excluding the reweighting task itself).
        """
        super(ReweightingCalculator, self).__init__()
        self.num_tasks = num_tasks
        self.softmax = nn.Softmax(dim=-1)  # Normalize logits into probability distribution

    def forward(self, task_logits):
        """
        Compute softmax-normalized loss weights.

        Args:
            task_logits (torch.Tensor): Logits from an external linear layer (batch_size, num_tasks).
        
        Returns:
            torch.Tensor: Loss weights (batch_size, num_tasks), where weights sum to 1.
        """
        return self.softmax(task_logits)


class MultiLossManager:
    """
    Handles multiple loss calculations, applying weights dynamically.
    """
    def __init__(self, loss_functions, do_softmax=False): 
        """
        Args:
            loss_functions (dict): A dictionary mapping task names to their corresponding loss functions.
        """
        self.loss_functions = loss_functions  # Dictionary: {"task_name": loss_function}
        num_tasks = len(loss_functions)
        self.do_softmax = do_softmax
        # self.reweighting_calculator = ReweightingCalculator(num_tasks)  # Initialize reweighting module

    def compute_weighted_loss(self, outputs, targets, task_logits):
        """
        Computes individual task losses and applies learned loss weights.

        Args:
            outputs (dict): A dictionary of task outputs. Format: {"task_name": tensor}.
            targets (dict): A dictionary of ground-truth targets. Format: {"task_name": tensor}.
            task_logits (torch.Tensor): Logits from a separate linear layer (batch_size, num_tasks).
        
        Returns:
            torch.Tensor: The final weighted loss value (scalar).
        """
        # Step 1: Compute task-specific losses
        task_losses = {task: self.loss_functions[task](outputs[task], targets[task])
                       for task in self.loss_functions}

        # Step 2: Compute task loss weights from logits
        if self.do_softmax: 
            loss_weights = torch.nn.functional.softmax(task_logits, dim=-1)
        else: 
            loss_weights = task_logits
        # loss_weights = self.reweighting_calculator(task_logits)  # Shape: (batch_size, num_tasks)

        # Step 3: Compute weighted loss
        weighted_loss = 0.0
        task_names = list(task_losses.keys())  # Ensure correct task ordering

        for i, task in enumerate(task_names):
            weighted_loss += loss_weights[:, i] * task_losses[task]  # Apply weights dynamically

        return weighted_loss.mean(), task_losses, loss_weights
