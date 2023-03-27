import warnings
from typing import Dict, List

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet50

from l5kit.environment import models


class RewardModel(nn.Module):

    def __init__(
        self,
        model_arch: str,
        num_input_channels: int,
        num_targets: int,
        criterion: nn.Module,
        pretrained: bool = True,
    ) -> None:
        """Initializes the planning model.

        :param model_arch: model architecture to use
        :param num_input_channels: number of input channels in raster
        :param num_targets: number of output targets
        :param weights_scaling: target weights for loss calculation
        :param criterion: loss function to use
        :param pretrained: whether to use pretrained weights
        """
        super().__init__()
        self.model_arch = model_arch
        self.num_input_channels = num_input_channels
        self.num_targets = num_targets
        self.criterion = criterion

        if pretrained and self.num_input_channels != 3:
            warnings.warn("There is no pre-trained model with num_in_channels != 3, first layer will be reset")

        if model_arch == "resnet18":
            self.model = resnet18(pretrained=pretrained)
            self.model.fc = nn.Linear(in_features=512, out_features=num_targets)
        elif model_arch == "resnet50":
            self.model = resnet50(pretrained=pretrained)
            self.model.fc = nn.Linear(in_features=2048, out_features=num_targets)
        elif model_arch == "simple_cnn":
            self.model = models.SimpleCNN_GN(self.num_input_channels, num_targets)
        else:
            raise NotImplementedError(f"Model arch {model_arch} unknown")

        if model_arch in {"resnet18", "resnet50"} and self.num_input_channels != 3:
            self.model.conv1 = nn.Conv2d(
                in_channels=self.num_input_channels,
                out_channels=64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # [batch_size, channels, height, width]
        image_batch = data_batch["image"]
        # [batch_size, num_steps * 2]
        outputs = self.model(image_batch)
        batch_size = len(data_batch["image"])

        if self.training:
            # if self.criterion is None:
            #     raise NotImplementedError("Loss function is undefined.")

            # # [batch_size, num_steps * 2]
            # targets = (torch.cat((data_batch["target_positions"], data_batch["target_yaws"]), dim=2)).view(
            #     batch_size, -1
            # )
            # # [batch_size, num_steps]
            # target_weights = (data_batch["target_availabilities"].unsqueeze(-1) * self.weights_scaling).view(
            #     batch_size, -1
            # )
            # loss = torch.mean(self.criterion(outputs, targets) * target_weights)
            loss_dict = batch_pairwise_loss(c_rewards=c_rewards, r_rewards=r_rewards,
                                        c_mask=mask['chosen'], r_mask=mask['rejected'],
                                        divergence_index=divergence_index)
            train_dict = {"loss": loss}
            return train_dict
        else:
            predicted = outputs.view(batch_size, -1, 3)
            # [batch_size, num_steps, 2->(XY)]
            pred_positions = predicted[:, :, :2]
            # [batch_size, num_steps, 1->(yaw)]
            pred_yaws = predicted[:, :, 2:3]
            eval_dict = {"positions": pred_positions, "yaws": pred_yaws}
            return eval_dict
def batch_pairwise_loss(c_rewards, r_rewards, c_mask, r_mask, divergence_index):
    """Calculates batch-style pairwise loss between rewards for chosen and rejected phrases.

    This loss is calculated in a batch-style (without looping) to speed up the code.
    """
    or_mask = torch.logical_or(c_mask, r_mask).type(torch.long)
    or_lengths = or_mask.sum(axis=1)
    or_indices = or_lengths - 1  # To gather the value at the last index of each row.

    d_rewards = (c_rewards - r_rewards)
    c_last_rewards = torch.gather(c_rewards, dim=1, index=or_indices.unsqueeze(-1))
    r_last_rewards = torch.gather(r_rewards, dim=1, index=or_indices.unsqueeze(-1))
    divergence_mask = batch_get_mask_equal_or_larger_than_indices(
        d_rewards, divergence_index
    )
    weights = divergence_mask * or_mask

    loss = -torch.log(torch.sigmoid(d_rewards)) * weights
    # Sum over each row first.
    loss = loss.sum(dim=-1)
    # Normalize row-wise using weights first.
    loss = loss / weights.sum(dim=-1)
    # Normalize with batch size.
    loss = loss.sum() / weights.shape[0]

    return {'loss': loss,
            'chosen_last_rewards': c_last_rewards.squeeze(-1),
            'rejected_last_rewards': r_last_rewards.squeeze(-1)}

def batch_get_mask_equal_or_larger_than_indices(matrix, indices):
    """Gets mask larger than indices in a batch fashion.

    Args:
      matrix: a 2D with [batch_size, dimension]
      indices: 1D index tensor, with values indicating the start of the threshold.
    Returns:
      A 2D matrix with mask of [0, 1], with 0 indicating values smaller than
        the index for each row.
    """
    assert len(matrix.shape) == 2
    batch_size, dim = matrix.shape

    assert len(indices.shape) == 1
    assert indices.shape[0] == batch_size

    # Turn indices into the same shape as matrix A.
    indices: torch.Tensor = indices.unsqueeze(1).repeat(1, dim)
    # Get a 2D matrix that goes from 0 to dim-1 for each row of batch_size.
    arange: torch.Tensor = torch.arange(dim).tile(batch_size, 1).to(matrix.device)
    # Calculate the mask
    return (arange >= indices).type(torch.long)

import torch
from torch.utils.data import Dataset, DataLoader
class PairwiseDataset(Dataset):
    """Dataset that yields each item as a dict of ['input_ids', 'mask']."""

    def __init__(self, dataset_type: str, tokenizer, max_length: int, split: str):
        """Initializes the dataset.

        Args:
            dataset_type: a str for type of dataset to load.
            # dict of ['chosen', 'rejected'], whose values are string.
            tokenizer: a tokenizer object.
            max_length: an int that is the maximum length for tokenizer's padding operation.
        """

        self._max_length = max_length
        self._split = split
        dataset_path = ''

        if dataset_type == 'CarperAI/openai_summarize_comparisons':
            self._dataset = load_dataset(dataset_path, split=split) # 
            self._process_fn = process_openai_summarize_comparisons
        else:
            raise ValueError(dataset_type)
    def 

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        sample = self._dataset[idx]
        processed = self._process_fn(sample,
                                     tokenizer=self._tokenizer,
                                     max_length=self._max_length)

        return processed

model = RewardModel(num_input_channels=5, num_targets=2)
ds = PairwiseDataset()
dl = DataLoader(ds, batch_size=4, shuffle=False)
batch = next(iter(dl))
output = model(**batch)