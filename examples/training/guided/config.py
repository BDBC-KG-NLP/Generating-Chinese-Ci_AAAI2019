#-*- coding: utf-8 -*-
import argparse

def str2bool(v):
    return v.lower() in ('true', '1')

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--vocab_dim', type=int, default=5956)
net_arg.add_argument('--char_emb_dim', type=int, default=100)
net_arg.add_argument('--char_hidden_dim', type=int, default=128)
net_arg.add_argument('--sentence_hidden_dim', type=int, default=200)
net_arg.add_argument('--latent_dim', type=int, default=24)

net_arg.add_argument('--tone_dim', type=int, default=3)
net_arg.add_argument('--vowel_dim', type=int, default=17)

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--is_train', type=str2bool, default=True)
train_arg.add_argument('--optimizer', type=str, default='adam')
train_arg.add_argument('--batch_size', type=int, default=5)
train_arg.add_argument('--validate_size', type=int, default=10)



train_arg.add_argument('--lr', type=float, default=0.001)



train_arg.add_argument('--beta1', type=float, default=0.25)
train_arg.add_argument('--beta2', type=float, default=0.999)
train_arg.add_argument('--weight_decay', type=float, default=0.0002)
train_arg.add_argument('--clip_norm', type=float, default=5)
train_arg.add_argument('--sample_times', type=int, default=10)
train_arg.add_argument('--using_first_sentence', type=str2bool, default=False)
train_arg.add_argument('--kl_min', type=float, default=0.5)
train_arg.add_argument('--kl_annealing', type=str2bool, default=True)
train_arg.add_argument('--char_dropout', type=float, default=0)
train_arg.add_argument('--max_step', type=int, default=50000)
train_arg.add_argument('--max_epochs', type=int, default=4)

train_arg.add_argument('--is_vowel_tone', type=str, default='ALL', choices=['NONE', 'ALL', 'YUNJIAO'])
train_arg.add_argument('--vowel_tone_input', type=str, default='DEC_RNN', choices=['NONE', 'DEC_RNN', 'PROJECTION', 'VOWEL_GRU_TONE_PROJECTION'])

train_arg.add_argument('--vowel_type', type=str, default='NEXT', choices=['CURRENT', 'NEXT'])
train_arg.add_argument('--tone_type', type=str, default='NEXT', choices=['CURRENT', 'NEXT'])

train_arg.add_argument('--train_same_as_test', type=str2bool, default=False)
train_arg.add_argument('--is_attention', type=str2bool, default=False)
train_arg.add_argument('--is_dec_embedding', type=str2bool, default=True)
train_arg.add_argument('--is_guided', type=str2bool, default=True)
train_arg.add_argument('--is_fixed_p', type=str2bool, default=True)
train_arg.add_argument('--is_tone_using_rule', type=str2bool, default=True)
# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--is_gpu', type=str2bool, default=True)
misc_arg.add_argument('--load_path', type=str, default='')
misc_arg.add_argument('--log_step', type=int, default=10)
misc_arg.add_argument('--save_step', type=int, default=1000)
misc_arg.add_argument('--num_log_samples', type=int, default=3)
misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--model_dir', type=str, default='saved_models')
misc_arg.add_argument('--data_dir', type=str, default='Data')

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
