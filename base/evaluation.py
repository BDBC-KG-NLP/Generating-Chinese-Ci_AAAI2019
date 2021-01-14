# -*- coding: UTF-8 -*-
import numpy as np
from utils import lookup
from functools import reduce
import rules
import copy
import pickle

known_punctuation = [4, 5, 541, 1705]
vocab_lookup, vowel_lookup, tone_lookup = lookup()

def write_down_x_hat(x_hat,output_path):
    with open(output_path,'a+') as f:
        for px in x_hat:
            f.write(''.join([vocab_lookup[_] for _ in px]))
            f.write('\n')
        f.write('\n\n')
    
    

def reconstruct_with_x_adv(x_hat , x,yunjiao,output_path):
    pack_yunjiao=[-1]*len(x_hat)
    for key, value in yunjiao.items():
        for v in value:
            pack_yunjiao[v] = key

    for px,gx,y in zip(x_hat,x,pack_yunjiao):
        print('====> 实际: {} 韵脚:{}'.format([vocab_lookup[_] for _ in gx],y))
        print('====> 预测: {}'.format([vocab_lookup[_] for _ in px]))
    write_down_x_hat(x_hat,output_path)   
    print('\n\n')

def reconstruct_with_x_adv_new(x_hat , x,yunjiao):
    pack_yunjiao=[-1]*len(x_hat)
    for key, value in yunjiao.items():
        for v in value:
            pack_yunjiao[v] = key
    ci = ''
    for px,gx,y in zip(x_hat,x,pack_yunjiao):
        #print('====> 预测: {}'.format([vocab_lookup[_] for _ in px]))
        ci += ''.join([vocab_lookup[_] for _ in px])+'\n'
    return ci


def reconstruct_with_x_abs(x_hat,x):
    with open('Eval_Data/Guided/mrcg_minus_abs.txt','a+') as f:
          for px in x:
              f.write(''.join([vocab_lookup[_] for _ in px]))
              f.write(' ')
          f.write('\n')
          for px in x_hat:
              f.write(''.join([vocab_lookup[_] for _ in px]))
              f.write(' ')
          f.write('\n\n')





def reconstruct_with_x_adv_adv(x_hat_beam, x_hat, x,yunjiao):
    pack_yunjiao=[-1]*len(x_hat)
    for key, value in yunjiao.items():
        for v in value:
            pack_yunjiao[v] = key

    for pb,px,gx,y in zip(x_hat_beam,x_hat,x,pack_yunjiao):

        # print('{:20s} 韵脚:{} | {:20s} | {:20s}'.format(''.join([vocab_lookup[_] for _ in gx]),y,''.join([vocab_lookup[_] for _ in px]),''.join([vocab_lookup[_] for _ in pb])))
        print('韵脚:{} | {:20s} | {:20s}'.format(y,''.join([vocab_lookup[_] for _ in px]),''.join([vocab_lookup[_] for _ in pb])))
    print('\n\n')


def print_yuanshi(xs):
    xs = xs.T
    print('-----------semantic reference:-----------\n')
    for x in xs:
        print(''.join([vocab_lookup[_] for _ in reversed(x) if _ not in [0,2]]))
    print('-----------------------------------------\n')



def reconstruct_with_x(x_hat , x):
    '''

    :param x_hat:
    :param x:
    :return:
    '''
    eos_token = 2
    for px, gx in zip(x_hat,x):

        l = np.where(gx == eos_token)[0][-1]
        gx = gx[:l]
        px = np.squeeze(px)[:l]
        print('====> 实际: {}'.format([vocab_lookup[_] for _ in gx]))
        print('====> 预测: {}'.format([vocab_lookup[_] for _ in px]))

def sample_helper(inputs, sentences=5):
    out=[]
    for x in inputs:
        out.append(x[:sentences])
    return np.array(out).T

def packing(flatten_x, flatten_x_hat, length_sequence, is_loose=False):
    _EOS = 2
    flatten_x = flatten_x.T
    flatten_x_hat = np.array(flatten_x_hat).T
    # print('flatten_x_hat',flatten_x_hat)
    counter = 0
    fs = copy.deepcopy(length_sequence['sentence'])
    indexes_x = [[fs.pop(0) for _ in range(ci_length)] for ci_length in length_sequence['ci']]
    indexes_x_hat = copy.deepcopy(indexes_x)
    for i in range(len(indexes_x)):
        for j in range(len(indexes_x[i])):
            # indexes_x[i][j] = flatten_x[counter][:indexes_x[i][j]-1]
            indexes_x[i][j] = flatten_x[counter][:indexes_x[i][j]]

            if not is_loose:
                
                temp = [id for id in flatten_x_hat[counter] if id != _EOS]
                k = 0
                for k,x in enumerate(temp):
                    if x in known_punctuation:
                        break
                indexes_x_hat[i][j] =temp[:k+1]
                x_len = len(indexes_x[i][j])
            else:
                indexes_x_hat[i][j] = [id for id in flatten_x_hat[counter]]
            counter += 1
    return indexes_x,indexes_x_hat


def eval_length_pingze_yunjiao(yunjiao_list,flatten_x,flatten_x_hat, length_sequence, is_debug= False):
    xs,x_hats = packing(flatten_x, flatten_x_hat, length_sequence)

    # print('xs,xhat',xs,x_hats)
    # print('yunjiao_list,flatten_x,flatten_x_hat',yunjiao_list,flatten_x,flatten_x_hat)
    l,ls,p,y = [], [], [], []
    # print('xs,xhats',len(xs),len(x_hats))
  #  if is_debug:
  #      reconstruct_with_x_adv(x_hats[0],xs[0], yunjiao_list[0])
    for x, x_hat, yunjiao in zip(xs, x_hats, yunjiao_list):
        l.append(rules.check_length(x, x_hat))
        ls.append(rules.check_length(x, x_hat,True))
        p.append(rules.check_pingze(x, x_hat,vocab_lookup, tone_lookup))
        y.append(rules.check_yunshe(yunjiao, x_hat, vocab_lookup, vowel_lookup,x))
    return sum(l)/len(l), sum(ls)/len(ls), sum(p)/len(p), sum(y)/len(y),eval_recovery(flatten_x, flatten_x_hat, length_sequence)

def eval_wrapper(flatten_x_train,flatten_x_beam,yunjiao_list,flatten_x,flatten_x_hat, length_sequence, is_debug= True):
    argmax = eval_length_pingze_yunjiao(yunjiao_list,flatten_x,flatten_x_hat, length_sequence)
    beam = eval_length_pingze_yunjiao(yunjiao_list,flatten_x,flatten_x_beam, length_sequence)
    train = eval_length_pingze_yunjiao(yunjiao_list,flatten_x,flatten_x_train, length_sequence)
    xs,x_hats = packing(flatten_x, flatten_x_hat, length_sequence)
    _,x_hat_beam = packing(flatten_x, flatten_x_beam, length_sequence)
    reconstruct_with_x_adv_adv(x_hat_beam[0],x_hats[0],xs[0], yunjiao_list[0])
    print('argmax',argmax)
    print('beam',beam)
    print('train',train)
    return argmax,beam
    # _,x_hats = packing(flatten_x, flatten_x_train, length_sequence)

def eval_wrapper_sr(flatten_x_beam,yunjiao_list,flatten_x,flatten_x_hat, length_sequence, is_debug= True):
    argmax = eval_length_pingze_yunjiao(yunjiao_list,flatten_x,flatten_x_hat, length_sequence)
    beam = eval_length_pingze_yunjiao(yunjiao_list,flatten_x,flatten_x_beam, length_sequence)

    xs,x_hats = packing(flatten_x, flatten_x_hat, length_sequence)
    _,x_hat_beam = packing(flatten_x, flatten_x_beam, length_sequence)
    reconstruct_with_x_adv_adv(x_hat_beam[0],x_hats[0],xs[0], yunjiao_list[0])
    # print('argmax',argmax)
    # print('beam',beam)
    return argmax,beam
    # _,x_hats = packing(flatten_x, flatten_x_train, length_sequence)

def eval_recovery(flatten_x, flatten_x_hat, length_sequence):
    xs,x_hats = packing(flatten_x, flatten_x_hat, length_sequence)
    r = []
    for x, x_hat in zip(xs, x_hats):
        r.append(rules.check_recovery(x, x_hat))
    return sum(r)/len(r)



