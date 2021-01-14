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
    prepare_dirs_and_logger(config)
    step = iterator.Iterator(config)
    trainer = Vrae(config, step)
    if config.is_train:
        trainer.train()
    else:
        trainer.different_semantic_different_z()

if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)
