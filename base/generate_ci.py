# 输入词牌生成词

import torch

from torch.autograd import Variable
from trainer import Vrae
from config import get_config
import iterator
from utils import prepare_dirs_and_logger
import time


def main(config,cipai):
    '''
    this main fun for ...
    :param config:
    :return:
    '''
    start = time.time()
    prepare_dirs_and_logger(config)# ci_generation\base\utils.py 日志输出
    step = iterator.Iterator(config) # without1/iterator.py
    trainer = Vrae(config, step)
    end = time.time()
    print('prepare:'+str(end-start))
    trainer.Generate_ci(cipai)
 #        trainer.Semantic_Latent_Relative()
if __name__ == "__main__":
    cipai = '且坐令'
    config, unparsed = get_config()#ci_generation\base\config.py
    print("gpu:"+str(config.is_gpu))
    main(config,cipai)
