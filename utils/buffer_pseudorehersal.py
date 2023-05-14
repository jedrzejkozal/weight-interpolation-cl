import torch

from typing import Tuple


class BufferPseudoRehersal:
    def __init__(self, device) -> None:
        self.device = device

        self.cumulative_mean: torch.TensorType = None
        self.M2: torch.TensorType = None
        self.cumulative_std: torch.TensorType = None
        self.n:int = 0

        self.image_size = None


    @torch.no_grad()
    def add_data(self, examples, labels=None) -> None:
        """
        Updates the data distribution statistics
        :param examples: tensor containing the images (with transforms already applied)
        :return:
        """
        if self.cumulative_mean == None:
            self.image_size = examples.shape[1:]
            self.cumulative_mean = torch.zeros(size=self.image_size, device=self.device)
        if self.M2 == None:
            self.M2 = torch.zeros(size=examples.shape[1:], device=self.device)

        # based on Welford's online algorithm for variance computation: 
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Weighted_incremental_algorithm
        for new_img in examples:
            self.n += 1
            delta = new_img - self.cumulative_mean
            self.cumulative_mean += delta / self.n
            delta2 = new_img - self.cumulative_mean
            self.M2 += delta * delta2

        cumulative_var = self.M2 / (self.n-1)
        self.cumulative_std = torch.sqrt(cumulative_var)
    

    @torch.no_grad()
    def get_data(self, size: int) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :return:
        """
        sample_size = [size] + list(self.image_size)
        sampled_images = torch.randn(size=sample_size, dtype=torch.float32, device=self.device) * self.cumulative_std + self.cumulative_mean
        labels = torch.zeros(size=[size], dtype=torch.long, device=self.device)
        return sampled_images.to(self.device), labels

    
    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        return self.n == 0
