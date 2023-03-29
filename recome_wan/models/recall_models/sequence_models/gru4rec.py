from typing import Dict
import torch
from torch import nn


class GRU4Rec(nn.Module):
    def __init__(self, enc_dict,config):
        super(GRU4Rec, self).__init__()
        self.enc_dict = enc_dict
        self.config = config
        self.embedding_dim = self.config['embedding_dim']
        self.max_length = self.config['max_length']
        self.num_layers = self.config.get('num_layers', 2)
        self.device = self.config['device']

        self.item_emb = nn.Embedding(self.enc_dict[self.config['item_col']]['vocab_size'], self.embedding_dim,
                                     padding_idx=0)

        self.gru = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.embedding_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bias=False
        )

        self.loss_fun = nn.CrossEntropyLoss()

        self.apply(self._init_weights)

    def calculate_loss(self, user_emb: torch.Tensor, pos_item: torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss for the model given a user embedding and positive item.

        Args:
        user_emb (torch.Tensor): A tensor representing the user embedding.
        pos_item (torch.Tensor): A tensor representing the positive item.

        Returns:
        The tensor representing the calculated loss value.
        """
        all_items = self.output_items()
        scores = torch.matmul(user_emb, all_items.transpose(1, 0))
        loss = self.loss_fun(scores, pos_item)
        return loss

    def gather_indexes(self, output: torch.Tensor, gather_index: torch.Tensor) -> torch.Tensor:
        """
        Gathers the vectors at the specific positions over a minibatch.

        Args:
        output (torch.Tensor): A tensor representing the output vectors.
        gather_index (torch.Tensor): A tensor representing the index vectors.

        Returns:
        The tensor representing the gathered output vectors.
        """
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def output_items(self) -> torch.Tensor:
        """
        Returns the item embedding layer weight.

        Returns:
        The tensor representing the item embedding layer weight.
        """
        return self.item_emb.weight

    def _init_weights(self, module: nn.Module):
        """
        Initializes the weight value for the given module.

        Args:
        module (nn.Module): The module whose weights need to be initialized.
        """
        if isinstance(module, nn.Embedding):
            torch.nn.init.kaiming_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight.data)

    def forward(self, data: Dict[str, torch.tensor], is_training: bool = True):
        """
        This method initializes the forward step to compute the user embeddings which will then be used for
        recommendations.

        Args:
            data (dict): a dictionary with input features as keys and the corresponding tensors as values .
            is_training (bool): a flag variable to set the mode of the model; default is True.

        Returns:
            dict: a dictionary with the user embeddings and model loss (if training) as keys and the corresponding
            tensors as values.
        """
        item_seq = data['hist_item_list']

        seq_emb = self.item_emb(item_seq)
        _, seq_emb = self.gru(seq_emb)
        user_emb = seq_emb[-1]
        if is_training:
            item = data['target_item'].squeeze()
            loss = self.calculate_loss(user_emb, item)
            output_dict = {
                'user_emb':user_emb,
                'loss':loss
            }
        else:
            output_dict = {
                'user_emb':user_emb
            }
        return output_dict

