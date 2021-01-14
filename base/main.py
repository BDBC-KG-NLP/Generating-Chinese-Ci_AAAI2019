import torch

from torch.autograd import Variable
from trainer import Vrae
from config import get_config
import iterator
from utils import prepare_dirs_and_logger


def main(config):
    '''
    this main fun for ...
    :param config:
    :return:
    '''
    prepare_dirs_and_logger(config)# ci_generation\base\utils.py 日志输出
    step = iterator.Iterator(config) # without1/iterator.py
    trainer = Vrae(config, step)
    if config.is_train:
        trainer.train()
    else:
        trainer.Metrical_Performances()
 #        trainer.Semantic_Latent_Relative()
if __name__ == "__main__":
    config, unparsed = get_config()#ci_generation\base\config.py
    print(config.is_gpu)
    main(config)
