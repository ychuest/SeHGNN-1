from model import * 
from algorithm import * 

from gh import * 
import genghao_lightning as gl 

# HG = pickle_load('/Dataset/Pickle/ACM_TransE.dglhg.pkl')
# METAPATH_LIST = [
#     ['pa', 'ap'],
#     ['pf', 'fp'], 
# ]

# HG = pickle_load('/Dataset/Pickle/IMDB.dglhg.pkl')
# METAPATH_LIST = [
#     ['md', 'dm'],
#     ['ma', 'am'], 
# ]

HG = pickle_load('/Dataset/Pickle/DBLP.dglhg.pkl')
METAPATH_LIST = [
    ['ap', 'pt', 'tp', 'pa'],
    ['ap', 'pc', 'cp', 'pa'],
]

HYPER_PARAM = dict(
    metapath_list = METAPATH_LIST,
    
    hidden_dim = 64, 
    
    num_epochs = 200,
    lr = 0.001,
    weight_decay = 0.,
)


def train_step(model, batch):
    label = batch['label']
    train_mask = batch['train_mask']
    
    pred = model()[train_mask]
    target = label[train_mask]

    return dict(pred=pred, target=target)


def val_step(model, batch):
    label = batch['label']
    val_mask = batch['val_mask']
    
    pred = model()[val_mask]
    target = label[val_mask]

    return dict(pred=pred, target=target)


def test_step(model, batch):
    label = batch['label']
    test_mask = batch['test_mask']
    
    pred = model()[test_mask]
    target = label[test_mask]

    return dict(pred=pred, target=target)


def main():
    hg_dataset = gl.HeteroGraphDataset(HG)
    loader = gl.SingleBatchDataLoader(
        dict(
            label = hg_dataset.label,
            train_mask = hg_dataset.train_mask,
            val_mask = hg_dataset.val_mask,
            test_mask = hg_dataset.test_mask,
        )
    )
    
    model = SeHGNN(
        in_dim = hg_dataset.feat_dim_dict[hg_dataset.infer_ntype],
        hidden_dim = HYPER_PARAM['hidden_dim'],
        out_dim = hg_dataset.num_classes,
        num_metapaths = len(HYPER_PARAM['metapath_list']), 
    )  
    
    model.pre_aggregate_neighbor(
        hg = hg_dataset.hg,
        infer_ntype = hg_dataset.infer_ntype,
        feat_dict = hg_dataset.feat_dict,
        metapath_list = HYPER_PARAM['metapath_list'],
    )  
    
    task = gl.MultiClassClassificationTask(
        model = model,
    )
    
    task.train_and_eval(
        train_dataloader = loader,
        val_dataloader = loader,
        test_dataloader = loader,
        train_step = train_step,
        val_step = val_step,
        test_step = test_step,
        optimizer_type = 'Adam',
        optimizer_param = dict(lr=HYPER_PARAM['lr'], weight_decay=HYPER_PARAM['weight_decay']),
        max_epochs = HYPER_PARAM['num_epochs'], 
    )


if __name__ == '__main__':
    main() 
