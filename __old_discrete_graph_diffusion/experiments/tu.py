from ..data_loaders.tu import load_data
from mate import mate
import ipdb

train_loader, test_loader = load_data(save_path=mate.save_dir)
ipdb.set_trace()
