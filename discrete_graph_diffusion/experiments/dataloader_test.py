from mate import mate
from ..data_loaders.qm9_p import QM9DataModule, QM9Infos, get_train_smiles
import os
import ipdb

remove_h = True
data_dir = os.path.join(mate.save_dir, "qm9/qm9_pyg/")
ipdb.set_trace()
datamodule = QM9DataModule(
    datadir=data_dir,
    train_batch_size=32,
    val_batch_size=32,
    test_batch_size=32,
    remove_h=remove_h,
)
dataset_infos = QM9Infos(
    datamodule=datamodule,
    remove_h=remove_h,
)
datamodule.prepare_data()
uba = next(iter(datamodule.train_dataloader()))
ipdb.set_trace()
