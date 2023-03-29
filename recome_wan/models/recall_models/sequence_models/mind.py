from typing import Dict
import torch
from torch import nn
from recome_wan.models.layers import CapsuleNetwork


class MIND(nn.Module):

    def __init__(self, enc_dict, config):
        super(MIND, self).__init__()
        self.enc_dict = enc_dict
        self.config = config
        self.embedding_dim = self.config['embedding_dim']
        self.max_length = self.config['max_length']
        self.device = self.config['device']

        self.item_emb = nn.Embedding(self.enc_dict[self.config['item_col']]['vocab_size'], self.embedding_dim,
                                     padding_idx=0)

        self.loss_fun = nn.CrossEntropyLoss()

        self.capsule = CapsuleNetwork(self.embedding_dim, self.max_length,
                                      bilinear_type=2, interest_num=self.config['K'])
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
        f"""
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
        mask = data['hist_mask_list']

        if is_training:
            item = data['target_item'].squeeze()
            seq_emb = self.item_emb(item_seq)  # Batch,Seq,Emb
            item_e = self.item_emb(item).squeeze(1)

            multi_interest_emb = self.capsule(seq_emb, mask, self.device)  # Batch,K,Emb

            cos_res = torch.bmm(multi_interest_emb, item_e.squeeze(1).unsqueeze(-1))
            k_index = torch.argmax(cos_res, dim=1)

            best_interest_emb = torch.rand(multi_interest_emb.shape[0], multi_interest_emb.shape[2]).to(self.device)
            for k in range(multi_interest_emb.shape[0]):
                best_interest_emb[k, :] = multi_interest_emb[k, k_index[k], :]

            loss = self.calculate_loss(best_interest_emb,item)
            output_dict = {
                'user_emb':multi_interest_emb,
                'loss':loss,
            }
        else:
            seq_emb = self.item_emb(item_seq)  # Batch,Seq,Emb
            multi_interest_emb = self.capsule(seq_emb, mask, self.device)  # Batch,K,Emb
            output_dict = {
                'user_emb': multi_interest_emb,
            }
        return output_dict


