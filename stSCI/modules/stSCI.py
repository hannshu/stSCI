import torch
from torch import nn
from typing import List, Tuple, Optional
from torch_geometric.data import Data
from .conv import fusion_conv


class stSCI(nn.Module):

    def __init__(
        self, 
        input_dim: int, 
        st_count: int,
        cluster_count: int,
        centroids: Tuple[torch.Tensor], 
        dims: List[int] = [512, 30],
        init_trans_matrix: Optional[torch.Tensor] = None,
    ) -> None:

        super().__init__()

        # encoder
        self.fusion_encoder = fusion_conv(input_dim, dims[0])
        self.encoder = nn.Linear(in_features=dims[0], out_features=dims[1])

        # decoder
        self.fusion_decoder = fusion_conv(dims[1], dims[0])
        self.decoder = nn.Linear(in_features=dims[0], out_features=input_dim)

        # misc
        self.recon_loss_func = nn.MSELoss()
        self.mnn_loss_func = nn.TripletMarginLoss()
        self.act_func = nn.ELU()
        self.softmax_func = nn.Softmax(dim=1)

        # centroids
        self.sc_centoids = nn.Parameter(centroids[0])
        self.st_centoids = nn.Parameter(centroids[1])

        if (init_trans_matrix):
            self.trans_matrix = nn.Parameter(init_trans_matrix)
        else:
            self.trans_matrix = nn.Parameter(torch.empty((st_count, cluster_count)))
            nn.init.kaiming_uniform_(self.trans_matrix, a=1.414)


    def forward(self, sc_data: torch.Tensor, st_data: Data, cluster_count: int = None) -> Tuple[torch.Tensor]:

        # encoder
        sc_embed, st_embed = self.fusion_encoder(sc_data, st_data.x, st_data.edge_index)
        sc_embed, st_embed = self.encoder(self.act_func(sc_embed)), self.encoder(self.act_func(st_embed))

        if (self.training):

            # decoder
            sc_recon, st_recon = self.fusion_decoder(sc_embed[: -cluster_count], st_embed, st_data.edge_index)
            sc_recon, st_recon = self.decoder(self.act_func(sc_recon)), self.decoder(self.act_func(st_recon))

            # generate DEC q distribution
            sc_q = self.get_q_distribution(self.sc_centoids, sc_embed[: -cluster_count])
            st_q = self.get_q_distribution(self.st_centoids, st_embed)

            return [sc_embed, st_embed], (sc_recon, st_recon), (sc_q, st_q)
        else:

            # decoder
            sc_recon, st_recon = self.fusion_decoder(sc_embed, st_embed, st_data.edge_index)
            sc_recon, st_recon = self.decoder(self.act_func(sc_recon)), self.decoder(self.act_func(st_recon))

            return (sc_embed, st_embed), (sc_recon, st_recon)


    def get_q_distribution(self, centroid: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:

        q = 1.0 / ((1.0 + torch.sum((embed.unsqueeze(dim=1) - centroid).pow(2), dim=2)) + 1e-6)
        q = q / torch.sum(q, dim=1, keepdim=True)

        return q


    def get_p_distribution(self, q: torch.Tensor) -> torch.Tensor:
            
        p = q.pow(2) / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)

        return p.detach()
    

    def kl_div(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sum(p * torch.log(p / (q + 1e-6)), dim=1))


    def loss(
        self, 
        sc_data: torch.Tensor, 
        st_data: torch.Tensor, 
        mnn_pairs: torch.Tensor, 
        recon: Tuple[torch.Tensor], 
        embed: Tuple[torch.Tensor],
        p: Tuple[torch.Tensor],
        q: Tuple[torch.Tensor],
        sc_cell_type: torch.Tensor,
    ) -> torch.Tensor:

        # reconstruction: sc_recon + st_recon + st coor recon
        recon_loss = self.recon_loss_func(sc_data, recon[0]) + self.recon_loss_func(st_data, recon[1])

        # symmetric mnn pairs (sc -> st) + (st -> sc)
        sc_index, st_index = mnn_pairs
        mnn_loss = self.mnn_loss_func(
            embed[0][sc_index], embed[1][st_index], 
            embed[0][torch.randint(embed[0].shape[0], sc_index.shape)]
        ) + self.mnn_loss_func(
            embed[1][st_index], embed[0][sc_index], 
            embed[1][torch.randint(embed[1].shape[0], st_index.shape)]
        )

        # transform matrix optimization
        cell_type_loss = self.recon_loss_func(
            embed[1], torch.mm(self.softmax_func(self.trans_matrix), sc_cell_type)
        )

        # DEC loss
        dec_loss = self.kl_div(p[0], q[0]) + self.kl_div(p[1], q[1])

        return recon_loss + mnn_loss + cell_type_loss + 0.01 * dec_loss
    