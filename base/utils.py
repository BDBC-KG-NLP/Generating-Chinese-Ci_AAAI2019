from __future__ import print_function
import toneVowelHelper
import os
import json
import logging
import ioHelper
from datetime import datetime

def prepare_dirs_and_logger(config):
    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    logger = logging.getLogger()

    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    # if config.load_path:
    #     if config.load_path.startswith(config.log_dir):
    #         config.model_dir = config.load_path
    #     else:
    #         if config.load_path.startswith(config.dataset):
    #             config.model_name = config.load_path
    #         else:
    #             config.model_name = "{}_{}".format(config.dataset, config.load_path)
    # else:
    #     config.model_name = "{}_{}".format(config.dataset, get_time())
    #
    # if not hasattr(config, 'model_dir'):
    #     config.model_dir = os.path.join(config.log_dir, config.model_name)
    # config.data_path = os.path.join(config.data_dir, config.dataset)
    model_sub_path = create_model_abbr(config)
    print('config.data_dir:',config.data_dir)
    print('model_sub_path:',model_sub_path)
    for path in [config.log_dir, config.data_dir, model_sub_path]:
        if not os.path.exists(path):
            os.makedirs(path)
    save_config(config)

def get_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def create_model_abbr(config):

    abbr = "%d_%d_%d_%d_%.3f_%s_%s_%s_%s" %(config.char_emb_dim,config.char_hidden_dim,
        config.sentence_hidden_dim,config.latent_dim,config.kl_min,config.is_vowel_tone,
        config.vowel_tone_input,config.vowel_type,config.tone_type)
    if config.train_same_as_test:
        extra = '_train_same_as_test'
        abbr += extra
    if config.is_attention:
        extra = '_is_attention'
        abbr += extra
    if config.is_dec_embedding:
        extra = '_is_dec_embedding'
        abbr += extra
    return os.path.join(config.model_dir, abbr)

def save_config(config):
    '''
    this function is aimed to save the config in json file
    '''
    param_path = os.path.join(create_model_abbr(config), "params.json")
    if not os.path.isfile(param_path):
        print("[*] MODEL dir: %s" % config.model_dir)
        print("[*] PARAM path: %s" % param_path)

        with open(param_path, 'w') as fp:
            json.dump(config.__dict__, fp, indent=4, sort_keys=True)

def buildVocab():
    path = os.getcwd()
    file_name = path + "/Data/vocab.txt"
    vocabDict = ioHelper.read(file_name)
    reverse = {}
    for i, key in enumerate(vocabDict):
        reverse[key] = i
    return vocabDict, reverse

def lookup():
    '''
    this function returns three dictionaries about
    vocabulary, vowel and tone
    '''
    inv, _ = buildVocab()
    vowel_lookup = toneVowelHelper.create_vowel_dict()
    tone_lookup = toneVowelHelper.create_tone_dict()
    return inv, vowel_lookup, tone_lookup


# a,b = buildVocab()
# print(b['_PAD'])
# print(b['_GO'])
# print(b['_EOS'])
# print(a[0])
