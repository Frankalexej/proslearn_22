import torch
import torch.nn as nn

class MultitaskingAdapter(nn.Module):
    def __init__(self, task_adapters):
        """
        A plug-in module for deriving task-specific representations.

        Args:
            task_adapters (dict): A dictionary of named task-specific layers.
        """
        super(MultitaskingAdapter, self).__init__()
        self.task_adapters = nn.ModuleDict(task_adapters)  # Store with names

    def forward(self, shared_representation, task_name):
        """
        Forward pass through the named task-specific adapter.

        Args:
            shared_representation (torch.Tensor): Input feature from the shared trunk.
            task_name (str): Name of the task-specific adapter to use.

        Returns:
            torch.Tensor: Task-specific output.
        """
        return self.task_adapters[task_name](shared_representation)