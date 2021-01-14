import rules
import ioHelper
import os
from utils import lookup


class Metric(object):
    def __init__(self):
        self._path = os.getcwd()
        # read raw dataset from the file
        self._data = ioHelper.load(self._path + "/Data/num.json")
        self._tone = ioHelper.load(self._path + "/Data/tone.json")
        self._vowel = ioHelper.load(self._path + "/Data/vowel.json")
        self._vocab = ioHelper.read(self._path + "/Data/vocab.txt")
        self._rules = ioHelper.load(self._path + "/Data/rules.json")
        self.vocab_lookup, self.vowel_lookup, self.tone_lookup = lookup()
        self.known_punctuation =  [4, 5, 541, 1705]
        
    def get_ref_yunjiao(self,key,index):
       # print(self._vowel[key])
        data = self._vowel[key][index]
        yunjiao_gt = rules.get_yunjiao_training(self._rules[key], self._data[key][index], self._tone[key][index], data)
        return yunjiao_gt
    def reconstruct_with_x_adv(self,x_hat , x,yunjiao):
        pack_yunjiao=[-1]*len(x_hat)
        for key, value in yunjiao.items():
            for v in value:
                pack_yunjiao[v] = key

        for px,gx,y in zip(x_hat,x,pack_yunjiao):
            print('====> 实际: {} 韵脚:{}'.format([self.vocab_lookup[_] for _ in gx],y))
            print('====> 预测: {}'.format([self.vocab_lookup[_] for _ in px]))
        print('\n\n')
    def get_metrics(self,key,index,x,x_hat,_EOS=2):
        
        for i in range(len(x)):
            temp = [id for id in x_hat[i] if id != _EOS]
            k = 0
            for k,j in enumerate(temp):
                if j in self.known_punctuation:
                    break
            x_hat[i]=temp[:k+1]

        yunjiao_gt = self.get_ref_yunjiao(key,index)
        length_sentence_level = rules.check_length(x, x_hat)
        length_ci_level = rules.check_length(x, x_hat,True)
        pingze = rules.check_pingze(x, x_hat,self.vocab_lookup, self.tone_lookup)
        yunshe = rules.check_yunshe(yunjiao_gt, x_hat, self.vocab_lookup, self.vowel_lookup,x)
        bow = rules.check_recovery(x, x_hat)
        self.reconstruct_with_x_adv(x_hat,x,yunjiao_gt)
        return length_sentence_level,length_ci_level,pingze,yunshe,bow


def test():
    key = '浣溪沙' 
    index = 4335
    x = [[107, 509, 281, 234, 386, 216,  13,   4], [1137,    6,  340,  415,   64,  286,   67,    4],[218,  23, 298, 441,   8, 342, 560,   4],[ 100,  402,  624,  624,  258,  124, 1060,    5],[751,   9, 185, 185, 842, 480, 114,   4],[606, 131,  28, 150, 206, 777, 511,   4]] 
    x_hat = [[149, 132, 480, 1482, 118, 10, 55, 5], [13, 2179, 154, 2799, 24, 7, 263, 4], [27, 175, 13, 76, 441, 162, 194, 4], [8, 338, 11, 334, 450, 347, 1115, 5], [11, 109, 28, 811, 543, 51, 690, 4], [9, 147, 651, 2020, 855, 513, 321, 4]]         
    m = Metric()
    length_sentence_level,length_ci_level,pingze,yunshe,bow = m.get_metrics(key,index,x,x_hat)

    print('length_sentence_level: ',length_sentence_level,'\n','length_ci_level: ',length_ci_level ,'\n','pingze: ',pingze,'\n','yunshe: ',yunshe,'\n','bow: ',bow,'\n')

test()
