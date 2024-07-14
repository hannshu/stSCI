from typing import Optional
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor, Tuple
from torch_geometric.utils import (
    contains_self_loops, 
    add_self_loops, 
    remove_self_loops, 
    softmax
)


class fusion_conv(MessagePassing):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs,
    ) -> None:
        
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.linear = Linear(in_channels, out_channels, bias=False, weight_initializer='glorot')
        self.att_src = Parameter(torch.empty(1, 1, out_channels))
        self.att_dst = Parameter(torch.empty(1, 1, out_channels))
        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()


    def reset_parameters(self) -> None:

        super().reset_parameters()

        self.linear.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        zeros(self.bias)

    
    def sc_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.bias
    

    def st_forward(self, x: torch.Tensor, edge_index: Adj) -> torch.Tensor:

        x_src = x_dst = self.linear(x).unsqueeze(dim=1)

        # calculate attention
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = (x_dst * self.att_dst).sum(dim=-1)
        alpha = (alpha_src, alpha_dst)

        # add self-loop if not
        if (not contains_self_loops(edge_index)):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.shape[0])

        # propagate
        alpha = self.edge_updater(edge_index, edge_attr=None, alpha=alpha)
        out = self.propagate(edge_index, x=(x_src, x_dst), alpha=alpha).mean(dim=1) + self.bias

        return out


    def forward(self, sc_x: torch.Tensor, st_x: torch.Tensor, st_edge_index: Adj) -> Tuple[torch.Tensor]:
        return self.sc_forward(sc_x), self.st_forward(st_x, st_edge_index)


    def edge_update(self, alpha_j: torch.Tensor, alpha_i: OptTensor,
                    edge_attr: OptTensor, index: torch.Tensor, ptr: OptTensor,
                    dim_size: Optional[int]) -> torch.Tensor:

        alpha = alpha_j + alpha_i
        alpha = F.sigmoid(alpha)
        alpha = softmax(alpha, index, ptr, dim_size)
        
        return alpha


    def message(self, x_j: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        return alpha.unsqueeze(-1) * x_j
