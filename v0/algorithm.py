from gh import * 
from sklearn.preprocessing import normalize


def normalize_matrix(mat: sp.csr_matrix) -> sp.csr_matrix:
    normalized_mat = normalize(mat, norm='l1', axis=1)
    
    return normalized_mat 


def aggregate_metapath_neighbors(
    hg: dgl.DGLHeteroGraph,
    metapath: list[str], 
    feat_dict: dict[NodeType, FloatTensor], 
) -> FloatArray:
    etypes = set(hg.canonical_etypes)
    etype_map = { etype[1]: etype for etype in etypes }
    src_ntype = etype_map[metapath[0]][0]
    dest_ntype = etype_map[metapath[-1]][2]
    assert src_ntype == dest_ntype
    feat = feat_dict[src_ntype].cpu().numpy()

    product = None 
    
    for etype in metapath:
        etype = etype_map[etype]
        
        adj_mat = hg.adj(etype=etype, scipy_fmt='csr').astype(np.float32)
        normalized_adj_mat = normalize_matrix(adj_mat)
        
        if product is None:
            product = normalized_adj_mat 
        else:
            product = product.dot(normalized_adj_mat)
            
    out = product.dot(feat)
    assert isinstance(out, ndarray) and out.dtype == np.float32 and out.shape == feat.shape 
    
    return out 
