from ..data_loaders.tu import TUDataset
from mate import mate
import ipdb

tu = TUDataset(
    root=mate.save_dir, name="ENZYMES", use_node_attr=True, use_edge_attr=True
)
ipdb.set_trace()
