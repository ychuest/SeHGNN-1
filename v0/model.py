from algorithm import * 

from gh import * 


class SeHGNN(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 num_metapaths: int):
        super().__init__()
        
        self.in_dim = in_dim 
        self.hidden_dim = hidden_dim 
        self.out_dim = out_dim 
        
        # 结点未经聚合的原始特征，作为一个特殊的元路径
        self.num_metapaths = num_metapaths + 1

        self.aggr_feat_list = [] 
        
        self.feature_projector_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, (in_dim + hidden_dim) // 2),
                nn.ReLU(),
                nn.Linear((in_dim + hidden_dim) // 2, hidden_dim),
            )
            for _ in range(self.num_metapaths)
        ])
        
        self.Q_fc = nn.Linear(hidden_dim, hidden_dim)
        self.K_fc = nn.Linear(hidden_dim, hidden_dim)
        self.V_fc = nn.Linear(hidden_dim, hidden_dim)
        self.beta = Parameter(torch.ones(1))
        
        self.final_projector = nn.Sequential(
            nn.Linear(self.num_metapaths * hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, out_dim), 
        )
        
    def pre_aggregate_neighbor(self,
                               hg: dgl.DGLHeteroGraph,
                               infer_ntype: NodeType, 
                               feat_dict: dict[NodeType, FloatTensor],
                               metapath_list: list[list[str]]):
        self.aggr_feat_list.clear()
        
        # 将未经聚合的初始特征加入
        raw_feat = feat_dict[infer_ntype].cpu().numpy()
        self.aggr_feat_list.append(raw_feat)
                               
        for metapath in metapath_list:
            aggr_feat = aggregate_metapath_neighbors(
                hg = hg,
                metapath = metapath,
                feat_dict = feat_dict, 
            )
            self.aggr_feat_list.append(aggr_feat)

    def forward(self) -> FloatTensor:
        # 1. Simplified Neighbor Aggregation
        assert len(self.aggr_feat_list) == self.num_metapaths
        num_nodes = len(self.aggr_feat_list[0])
        
        # 2. Multi-layer Feature Projection
        assert len(self.aggr_feat_list) == len(self.feature_projector_list)
        
        h_list = [
            proj(
                torch.from_numpy(feat).to(self.device)
            )
            for feat, proj in zip(self.aggr_feat_list, self.feature_projector_list)
        ]
        
        h = torch.stack(h_list)
        assert h.shape == (self.num_metapaths, num_nodes, self.hidden_dim)
        
        # 3. Transformer-based Semantic Aggregation
        h = h.transpose(0, 1)
        assert h.shape == (num_nodes, self.num_metapaths, self.hidden_dim)
        
        Q = self.Q_fc(h)
        K = self.K_fc(h)
        V = self.V_fc(h)
        assert Q.shape == K.shape == V.shape == (num_nodes, self.num_metapaths, self.hidden_dim)
        
        attn = Q @ (K.transpose(1, 2))
        assert attn.shape == (num_nodes, self.num_metapaths, self.num_metapaths)

        attn = torch.softmax(attn, dim=-1)
        
        attn_out = self.beta * (attn @ h) + h 
        assert attn_out.shape == (num_nodes, self.num_metapaths, self.hidden_dim)
        
        attn_out = attn_out.view(num_nodes, self.num_metapaths * self.hidden_dim)

        out = self.final_projector(attn_out)
        assert out.shape == (num_nodes, self.out_dim)

        return out 
