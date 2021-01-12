# -*- coding: utf-8 -*-
import os
import ioHelper
import userHelper
import random
import copy
import numpy as np
import rules
import config
import statistics_hardcoded
from argparse import Namespace
from typing import *

def tran(a):
    row = len(a)
    col = len(a[0])
    # print(col,row)
    b = []
    for x in range(col):
        c = []
        for y in range(row):
            c.append(a[y][x])
        b.append(c)
    return b

class Iterator(object):
    def __init__(self, config:Namespace) -> None:
        self._path = os.getcwd()
        self._step = 0
        self._epoch = 0

        # set value from config
        self._batchSize = config.batch_size
        self._validateSize = config.validate_size
        self._is_train = config.is_train
        self._is_vowel_tone = config.is_vowel_tone

        self._vowel_type = config.vowel_type
        self._tone_type = config.tone_type

        self._is_tone_using_rule = config.is_tone_using_rule

        # read raw dataset from the file
        self._data = ioHelper.load(self._path + "/Data/num.json")
        self._tone = ioHelper.load(self._path + "/Data/tone.json")
        self._vowel = ioHelper.load(self._path + "/Data/vowel.json")
        self._vocab = ioHelper.read(self._path + "/Data/vocab.txt")

        # when use end-of-sentence vowel
        if self._is_vowel_tone != 'abc':
            self._rules = ioHelper.load(self._path + "/Data/rules.json")
            self._cir_chars = ioHelper.load(self._path +'/Data/sim.json')
            self._cuml = rules.yunshe_statistics(self._vowel)
        if self._is_train:
            self.setupTrain()
            self.setupTest()
        else:
            self.setupALL()

    def setupTrain(self) -> None:
        r"""set up training envirnoment

        inputs: None

        outputs: None

        notes:
            - **tune_list**: list of tune name
            - **data_list**: list of data index for a given tune

        """
        # create a new key list for the drawing
        key_list = list(self._data.keys())

        holder = {}
        for i in key_list:
            length = len(self._data[i])
            # create a fresh indexing for the drawing
            # train index does not include the last one
            holder[i] = [i for i in range(length - 1)]
            # shuffle the index for each elements under key
            random.shuffle(holder[i])
        # shuffle the key list
        random.shuffle(key_list)

        self._key_list = key_list
        self._key_holder = holder

    def setupALL(self) -> None:
        r"""set up training envirnoment

        inputs: None

        outputs: None

        notes:
            - **tune_list**: list of tune name
            - **data_list**: list of data index for a given tune

        """
        # create a new key list for the drawing
        key_list = list(self._data.keys())
        # print(key_list)
        holder = {}
        for i in key_list:
            length = len(self._data[i])
            holder[i] = [i for i in range(length)]
        self._key_list = key_list
        self._key_holder = holder

    # FIXME   should have mode switching instead of using same iterator
    #         dataset should be shared across iterators to relief the
    #         memory overhead when having multiple iterators
    def setupTest(self) -> None:
        r"""set up test envirnoment

        inputs: None

        outputs: None

        notes:
            - tune_list: list of tune name
            - data_list: list of data index for a given tune

        """
        key_list = list(self._data.keys())
        test = {}
        for i in key_list:
            #test index include only the last one of each key
            length = len(self._data[i])
            test[i] = length - 1
        random.shuffle(key_list)
        self._test_key = key_list
        self._test_holder = test


    # pad encoder sentence data
    def padEncode(self, data:List[List[int]]) -> List[List[int]]:
        r"""pad encoder data

        inputs: input_data
             -**input_data**: a list of list with size [number of sentence], [length per sentence]

        outputs: padded_encoder_data
            -**padded_encoder_data**: a matrix with size [number of sentence, length of sentence]

        notes:
            - for each pad, format is reverse of orginal sentence with EOS at front

        """
        result = copy.deepcopy(data)
        for i in range(len(result)):
            #reverse the each line of data and add EOS at front
            result[i] = [2] + result[i][::-1]
        return result


    # pad decoder sentence data
    def padDecode(self, data:List[List[int]]) -> List[List[int]]:
        r"""pad decoder data

        inputs: input_data
             -**input_data**: a list of list with size [number of sentence], [length per sentence]

        outputs: padded_decoder_data
            -**padded_decoder_data**: a matrix with size [number of sentence, length of sentence]

        notes:
            - for each pad, format is orginal sentence with GO at front

        """
        result = copy.deepcopy(data)
        for i in range(len(result)):
            #add GO at front for each line of data
            result[i] = [1] + result[i]
        return result


    # pad result(reference) sentence data
    def padResult(self, data:List[List[int]]) -> List[List[int]]:
        r"""pad result data

        inputs: input_data
             -**input_data**: a list of list with size [number of sentence], [length per sentence]

        outputs: padded_result_data
            -**padded_result_data**: a matrix with size [number of sentence, length of sentence]

        notes:
            - for each pad, format is orginal sentence with EOS at end

        """

        result = copy.deepcopy(data)
        for i in range(len(result)):
            #add EOS at end for each line of data
            result[i] = result[i] + [2]
        return result


    # pad tone {0,1,2}
    def padTone(self, key:int, index:int) -> List[List[int]]:
        r"""pad tone data

        inputs: tone_data
             -**tone_data**: a list of list with size [number of sentence], [length per sentence]

        outputs: padded_tone_data
            -**padded_tone_data**: a matrix with size [number of sentence, length of sentence]

        notes:
            - for each pad, format is orginal sentence with no sound at end

        """
        data = rules.get_tone_by_rule(self._rules[key], self._data[key][index], self._tone[key][index],self._is_tone_using_rule)
        result = copy.deepcopy(data)
        for i in range(len(result)):
            if self._tone_type == 'NEXT':
                result[i] = result[i] + [0]
            else:
                result[i] = [0] + result[i]
        return result

        


    # pad vowel (0 ~ 16)
    def padVowel(self, key:int, index:int) -> List[List[int]]:
        is_use_yun_stat = True
        self._is_train = True
        data = self._vowel[key][index]
        result = copy.deepcopy(data)
        yunjiao = rules.get_yunjiao_training(self._rules[key], self._data[key][index], self._tone[key][index], data)
        # if self._is_vowel_tone == 'YUNJIAO':
        #
        #     flag = False
        #     if not self._is_train:
        #         yunjiao = rules.random_yunjiao(yunjiao,self._cuml)
        #     for i in range(len(result)):
        #         for key in yunjiao:
        #             if i in yunjiao[key]:
        #                 flag = True
        #                 if self._vowel_type == 'NEXT':
        #                     result[i] = [16]* (len(result[i]) - 1) + [key] + [16]
        #                 else:
        #                     result[i] = [16]* (len(result[i])) +[key]
        #         if flag == False:
        #             result[i] = [16]* (len(result[i])+1)
        #         flag = False
        # else:
        #     for i in range(len(result)):
        #
        #         if self._vowel_type == 'NEXT':
        #             result[i] = result[i] + [16]
        #
        #         else:
        #             result[i] = [16] + result[i]



        if not self._is_train:
            yunjiao = rules.random_yunjiao(yunjiao,self._cuml)
            for i in range(len(result)):
                flag = False
                for key in yunjiao:
                    if i in yunjiao[key]:
                        result[i] = [17]*(len(result[i]) - 2) + [key] + [16]*2
                        flag = True
                if not flag:
                    result[i] = [17] * len(result[i]) + [16]
        else:
            if is_use_yun_stat:
                for i in range(len(result)):
                    flag = False
                    for key in yunjiao:
                        if i in yunjiao[key]:
                            result[i] = [17]*(len(result[i])- 2) + [key] + [16]*2
                            flag = True
                    if not flag:
                        result[i] = [17] * (len(result[i])-1) + [16]*2
            else:
                for i in range(len(result)):
                    result[i] = result[i] + [16]

                    
        return result, yunjiao

    def padVowel_test(self, key:int, index:int) -> List[List[int]]:
        yun_stat_onehot= [0.03555043186137205, 0.1400235970134467, 0.063320694220533, 0.032121839846529346, 
                  0.08284893408147274, 0.027628680278480786, 0.07843070854085796, 0.05577271212544244, 
                  0.024786591132206322, 0.09202871270185804, 0.028394952760940638, 0.009467265898712973, 
                  0.08457818086840613, 0.07266150894795147, 0.11503544055263845, 0.057349749169150956]

        data = self._vowel[key][index]
        result = copy.deepcopy(data)
        result_gt = copy.deepcopy(data)
        # print(key)
        yunjiao_gt = rules.get_yunjiao_training(self._rules[key], self._data[key][index], self._tone[key][index], data)
        yunjiao = rules.random_yunjiao(yunjiao_gt,self._cuml)
        ranodm_n = 2
        for i in range(len(result)):
            
            flag = False
            for key in yunjiao:
                if i in yunjiao[key]:
                    # result[i] = [17]*(len(result[i]) - 2) + np.random.choice([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],(1,ranodm_n),replace=True,p=yun_stat_onehot).tolist() + [16]*2
                    # result[i] = result[i] + [16]
                    # result[i] = [result[i][0]]+[17]*(len(result[i]) - 3) + [key] + [16]*2
                    result[i] =  np.random.choice([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],(len(result[i]) - 2,ranodm_n),replace=True,p=yun_stat_onehot).tolist()+ [key] + [16]*2
                    # print('lalalal',result[i])
                    # result[i] = [np.random.choice([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],1, p=yun_stat_onehot)[0]] +[17]*(len(result[i]) - 3)+ [key] + [16]*2
                    flag = True
                    # print('result[i]',i,result[i])
            if not flag:
                # result[i] = [17] * (len(result[i])-1)+ [16]*2
                # result[i] = [np.random.choice([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],1, p=yun_stat_onehot)[0]] +[17]*(len(result[i]) - 3)+ [np.random.choice([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],1, p=yun_stat_onehot)[0]] + [16]*2
                result[i] =  np.random.choice([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],(len(result[i]) - 1,ranodm_n),replace=True,p=yun_stat_onehot).tolist()+  [16]*2
                # print('result[i]',i,result[i])
            # result[i] = result[i] + [16]
    
        for i in range(len(result_gt)):
            result_gt[i] = result_gt[i] + [16]

                    
        return result,result_gt, yunjiao,yunjiao_gt

    def padSentence(self,lenArray:List[int]) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
        r"""pad sentence to [B,max(C)] then to [max(c),B] both for forward and backward sentence seq,
             as well as a flatten 1d list of valid index

           Notations:
               - C: 1D list containing ci length sequence over the batch. (在一批宋词中每首词的诗长，句数)
               — max(C): max ci length sequence over the batch. (在一'批'宋词中最大词长数(在一批最多句数的词))
               - B: batch size. (一批宋词的数量(多少首))

        out : Sentenceindex,B

        """
        # selectIndex: flatten out and only select index where !=padToken

        out = []
        out_reverse = []
        maxLen = max(lenArray)
        padToken = -1
        cumsum = np.cumsum([0]+lenArray)

        for i in range(len(lenArray)):
            pad_num = maxLen - lenArray[i]
            out.append([j for j in range(cumsum[i], cumsum[i + 1])] + [padToken] * pad_num)
            out_reverse.append([j for j in range(cumsum[i], cumsum[i + 1])][::-1] + [padToken] * pad_num)

        out = np.array(out).T
        out_reverse = np.array(out_reverse).T
        selectIndex = np.where(out.flatten()!= padToken)[0]

        return out, out_reverse, selectIndex.tolist()


    def pad(self, metaData:Dict[str, List]) -> Dict[str, List]:
        r"""pad meta data

        inputs: meta_data
             -**data_data**: a collection of data

        outputs: padded_meta_data
            -**padded_meta_data**: a collection of padded data

        """
        length = len(max(metaData['decode_data'], key = len))
        # length = 9
        key = ['encode_data', 'result_data', 'decode_data', 'vowel', 'tone']
        metaHolder = {}
        for i in key:
            metaHolder[i] = []
            for j in metaData[i]:
                diff = length - len(j)
                for k in range(diff):
                    #pad append num, current -1, change if needed
                    j.append(0)
                metaHolder[i].append(j)

            if self._is_train or i != 'vowel': # the negation of the exp is if is_testing and is padding vowel, we can not use numpy.array, we have to use own transpose method.
                metaHolder[i] = np.array(metaHolder[i])

        return metaHolder
    def pad_test(self, metaData:Dict[str, List]) -> Dict[str, List]:
        r"""pad meta data

        inputs: meta_data
             -**data_data**: a collection of data

        outputs: padded_meta_data
            -**padded_meta_data**: a collection of padded data

        """
        length = len(max(metaData['decode_data'], key = len))
        key = ['encode_data', 'result_data', 'decode_data', 'vowel', 'tone', 'vowel_ground_truth']
        metaHolder = {}
        for i in key:
            metaHolder[i] = []
            for j in metaData[i]:
                diff = length - len(j)
                for k in range(diff):
                    #pad append num, current -1, change if needed
                    j.append(0)
                metaHolder[i].append(j)
            # if i != 'vowel':
            #     metaHolder[i] = np.array(metaHolder[i])
            # # metaHolder[i] = np.array(metaHolder[i])
            if self._is_train or i != 'vowel': # the negation of the exp is if is_testing and is padding vowel, we can not use numpy.array, we have to use own transpose method.
                metaHolder[i] = np.array(metaHolder[i])
        return metaHolder
    def pad_test_semantic_rule(self, metaData:Dict[str, List]) -> Dict[str, List]:
        r"""pad meta data

        inputs: meta_data
             -**data_data**: a collection of data

        outputs: padded_meta_data
            -**padded_meta_data**: a collection of padded data

        """
        
        key = ['encode_data', 'result_data', 'decode_data', 'vowel', 'tone', 'vowel_ground_truth']
        metaHolder = {}
        for i in key:
            if i == 'encode_data':
                length = len(max(metaData['encode_data'], key = len))
            else:
                length = len(max(metaData['decode_data'], key = len))
            metaHolder[i] = []
            for j in metaData[i]:
                diff = length - len(j)
                for k in range(diff):
                    #pad append num, current -1, change if needed
                    j.append(0)
                metaHolder[i].append(j)
            metaHolder[i] = np.array(metaHolder[i])
        return metaHolder
    def mask(self, metaData:Dict[str, List]) -> List[List[int]]:
        r"""generate mask

        inputs: meta_data
             -**data_data**: a collection of data

        outputs: mask_data
            -**mask_data**: a mask for the result data

        notes:
            - 1 if input data elment is non zero

        """

        mask = []
        for j in metaData['result_data']:
            mask.append([0 if i == 0 else 1 for i in j])
        return np.array(mask, dtype=np.float32)



    # create: draw index from random selected tune, remove tune when empty
    def drawKey(self, is_train:bool) -> Tuple[int, int]:
        r"""draw key

        inputs: is_train
             -**is_train**: a bool to flag to draw from train or test

        outputs: tune_index, data_index
            -**tune_index**: an index of tune
            -**data_index**: an index of data given tune index

        """

        # when train flag is used
        if is_train:
            # shuffle and grab current key list
            random.shuffle(self._key_list)
            current_key = self._key_list[0]

            # when key list is not empty and current key is empty
            while len(self._key_holder[current_key]) == 0 and len(self._key_list) != 0:
                # grap new key from the list and shuffle, remove the empty key
                self._key_list.remove(current_key)
                random.shuffle(self._key_list)
                # refill the key list if the key is running out, else draw index from key list
                if len(self._key_list) != 0:
                    current_key = self._key_list[0]
                else:
                    self.setupTrain()
                    # increase counting when all key running out
                    self._epoch = self._epoch + 1
                    current_key = self._key_list[0]

            current_index = self._key_holder[current_key].pop()

        else:
            # test flag is much simplier
            # just pop key from key list
            # refill when needed
            if len(self._test_key) != 0:
                current_key = self._test_key.pop()
            else:
                self.setupTest()
                self._epoch = self._epoch + 1
                current_key = self._key_list.pop()
            current_index = self._test_holder[current_key]
        # print(current_key,current_index)
        return current_key, current_index

    def iterativeKey(self):


        current_key = self._key_list[0]

        # when key list is not empty and current key is empty
        while len(self._key_holder[current_key]) == 0 and len(self._key_list) != 0:
            # grap new key from the list and shuffle, remove the empty key
            self._key_list.remove(current_key)
            current_key = self._key_list[0]

        current_index = self._key_holder[current_key].pop()
        # print(current_key,current_index)
        if len(self._key_list) == 0 and len(self._key_holder[current_key]) ==0:
            self._epoch = self._epoch + 1
        return current_key, current_index

    def draw_key_from_piror(self,is_semantic=True):
        draw = random.choice(range(statistics_hardcoded.cipai_cuml[-1][1]))
        if is_semantic: 
            c = '语义: '
        else:
            c = '规则: '
        for x in statistics_hardcoded.cipai_cuml:
            if draw < x[1]:
                key = x[0]
                print(c+key+'\n')
                break

        index = random.choice(range(len(self._data[key])))
        if is_semantic:
            # temp = ''
            print('*******************************')
            for x in self._cir_chars[key][index]:
                # temp += ''.join(x) 
                print(''.join(x))
            print('*******************************')
            # print(temp)
        return key,index

    # def fix_key(self,is_semantic=True):
    #     # draw = random.choice(range(statistics_hardcoded.cipai_cuml[-1][1]))
    #     if is_semantic: 
    #         c = '语义: '
    #         key = '浣溪沙'
    #     else:
    #         c = '规则: '
    #         key = '长相思'
            
    #     print(c+key+'\n')   
    #     index = random.choice(range(len(self._data[key])))
    #     if is_semantic:
    #         print('*******************************')
    #         for x in self._cir_chars[key][index]:
    #             # temp += ''.join(x) 
    #             print(''.join(x))
    #         print('*******************************')
        # return key,index

    # def fix_key(self,key,is_semantic=True):
    #     # draw = random.choice(range(statistics_hardcoded.cipai_cuml[-1][1]))
    #     if is_semantic: 
    #         c = '语义: '
           
    #     else:
    #         c = '规则: '
    #         index = random.choice(range(len(self._data[key])))
        
    #     print(c+key+'\n') 
        
        
    #     if is_semantic:
    #         index = 6
    #         print('*******************************')
    #         for x in self._cir_chars[key][index]:
    #             # temp += ''.join(x) 
    #             print(''.join(x))
    #         print('*******************************')
    #     return key,index

    def fix_key(self,key,is_semantic=True):
        draw = random.choice(range(statistics_hardcoded.cipai_cuml[-1][1]))
        
        index = random.choice(range(len(self._data[key[0]])))
        
        while(len(self._data[key[0]][index])!=key[1]):
            index = random.choice(range(len(self._data[key[0]])))
        # print(key[0],len(self._data[key[0]][index]))
        # print(key[0],index)

        # if is_semantic:
        #     index = 1813
        # else:
        #     index =405
        # print(key[0],index)
        return key[0],index
    # function to get user intent and return most relevant data
    def evaluation(self) -> Tuple[Dict[str, List], List, List]:
        r"""evaluation

        inputs: None

        outputs: data_list, length_list, vocal_list

        notes:
            - same return type like next, but affected by user instead random draw
        """

        correct_p_index, correct_t_index = self.get_user_intent()
        # FIXME need a flag to switch between most match to match tune
        key, index = correct_t_index

        metaData = {}
        ci_length_sequence = []
        metaData['encode_data'], metaData['decode_data'], metaData['result_data'], \
        metaData['tone'], metaData['vowel'] = [], [], [], [], []

        # grab all the data type from the data set
        # FIXME only encoder type and vocal information is needed
        encodeData = self.padEncode(self._data[key][index])
        ci_length_sequence.append(len(encodeData))
        decodeData = self.padDecode(self._data[key][index])
        resultData = self.padResult(self._data[key][index])
        tone = self.padTone(key,index)
        vowel, yunjiao = self.padVowel(key, index)

        metaData['encode_data'] = encodeData
        metaData['decode_data'] = decodeData
        metaData['result_data'] = resultData
        metaData['tone'] = tone
        metaData['vowel'] = vowel

        # pad the sentence level
        # since batch one, no actual padding will happen
        sentence_length_sequence = self.getSequence(metaData['encode_data'])
        padded_char = self.pad(metaData)
        for k,v in padded_char.items():
            padded_char[k] = v.T
        
        sentencePadding, sentencePadding_reverse, selectIndex = self.padSentence(ci_length_sequence)
        padded_sentence = {'forward':sentencePadding, 'backward':sentencePadding_reverse, 'valid_index':selectIndex }
        length_sequence = {'ci': ci_length_sequence,'sentence': sentence_length_sequence}

        return padded_char, padded_sentence, length_sequence, [yunjiao]


    # key next function to generate next batch
    def next(self, is_train:bool=True) -> Tuple[Dict[str, List], Dict[str, List], List]:
        r"""next

        inputs: is_train
            - **is_train**: boolean flag to decide to activate train

        outputs: data_list, length_list, vocal_list

        notes:
            - same return type like evaluation, but affected by user instead random draw

        """

        metaData = {}
        metaData['encode_data'], metaData['decode_data'], metaData['result_data'],\
        metaData['tone'], metaData['vowel'] = [],[],[],[],[]
        ci_length_sequence = []
        yunjiao_list = []

        batchSize = self._batchSize if is_train else self._validateSize

        # FIXME should have better way than this loop and collection
        for i in range(batchSize):
            key, index = self.drawKey(is_train)
            # key, index = self.iterativeKey()
            encodeData = self.padEncode(self._data[key][index])
            decodeData = self.padDecode(self._data[key][index])
            resultData = self.padResult(self._data[key][index])
            tone = self.padTone(key,index)
            vowel, yunjiao = self.padVowel(key, index)
            yunjiao_list.append(yunjiao)
            ci_length_sequence.append(len(encodeData))
            metaData['encode_data'] = metaData['encode_data'] + encodeData
            metaData['decode_data'] = metaData['decode_data'] + decodeData
            metaData['result_data'] = metaData['result_data'] + resultData
            metaData['tone'] = metaData['tone'] + tone
            metaData['vowel'] = metaData['vowel'] + vowel

        sentence_length_sequence = self.getSequence(metaData['encode_data'])

        padded_char = self.pad(metaData)

        for k,v in padded_char.items():
            padded_char[k] = v.T
        mask = self.mask(padded_char)

        if is_train:
            self._step = self._step + 1

        sentencePadding, sentencePadding_reverse, selectIndex = self.padSentence(ci_length_sequence)
        padded_sentence = {'forward':sentencePadding, 'backward':sentencePadding_reverse, 'valid_index':selectIndex }
        length_sequence = {'ci': ci_length_sequence,'sentence': sentence_length_sequence}

        return padded_char, padded_sentence, mask, length_sequence, yunjiao_list

    def next_for_visualization(self):


        metaData = {}
        metaData['encode_data'], metaData['decode_data'], metaData['result_data'],\
        metaData['tone'], metaData['vowel'] = [],[],[],[],[]
        ci_length_sequence = []
        yunjiao_list = []
        key, index = self.iterativeKey()
        
        encodeData = self.padEncode(self._data[key][index])
        decodeData = self.padDecode(self._data[key][index])
        resultData = self.padResult(self._data[key][index])
        tone = self.padTone(key,index)
        vowel, yunjiao = self.padVowel(key, index)
        yunjiao_list.append(yunjiao)
        ci_length_sequence.append(len(encodeData))
        metaData['encode_data'] = metaData['encode_data'] + encodeData
        metaData['decode_data'] = metaData['decode_data'] + decodeData
        metaData['result_data'] = metaData['result_data'] + resultData
        metaData['tone'] = metaData['tone'] + tone
        metaData['vowel'] = metaData['vowel'] + vowel

        sentence_length_sequence = self.getSequence(metaData['encode_data'])

        padded_char = self.pad(metaData)

        for k,v in padded_char.items():
            padded_char[k] = v.T
        mask = self.mask(padded_char)

        sentencePadding, sentencePadding_reverse, selectIndex = self.padSentence(ci_length_sequence)
        padded_sentence = {'forward':sentencePadding, 'backward':sentencePadding_reverse, 'valid_index':selectIndex }
        length_sequence = {'ci': ci_length_sequence,'sentence': sentence_length_sequence}

        return padded_char, padded_sentence, mask, length_sequence, yunjiao_list

    # key next function to generate next batch
    def next_test(self, is_train:bool=True,is_semantic:bool=True) -> Tuple[Dict[str, List], Dict[str, List], List]:
        r"""next

        inputs: is_train
            - **is_train**: boolean flag to decide to activate train

        outputs: data_list, length_list, vocal_list

        notes:
            - same return type like evaluation, but affected by user instead random draw

        """

        metaData = {}
        metaData['encode_data'], metaData['decode_data'], metaData['result_data'],\
        metaData['tone'], metaData['vowel'], metaData['vowel_ground_truth'] = [],[],[],[],[],[]
        ci_length_sequence = []
        yunjiao_list = []
        yunjiao_gt_list= []

        batchSize = self._batchSize if is_train else self._validateSize

        # FIXME should have better way than this loop and collection
        for i in range(batchSize):
            # key, index = self.drawKey(is_train)
            # key, index = self.fix_key(is_semantic)
            key, index = self.iterativeKey()
            # key, index = self.draw_key_from_piror(is_semantic)
            encodeData = self.padEncode(self._data[key][index])
            decodeData = self.padDecode(self._data[key][index])
            resultData = self.padResult(self._data[key][index])
            tone = self.padTone(key,index)
            vowel,vowel_ground_truth,yunjiao,yunjiao_ground_truth = self.padVowel_test(key, index)

            yunjiao_gt_list.append(yunjiao_ground_truth)
            yunjiao_list.append(yunjiao)
            ci_length_sequence.append(len(encodeData))
            metaData['encode_data'] = metaData['encode_data'] + encodeData
            metaData['decode_data'] = metaData['decode_data'] + decodeData
            metaData['result_data'] = metaData['result_data'] + resultData
            metaData['tone'] = metaData['tone'] + tone
            metaData['vowel'] = metaData['vowel'] + vowel
            metaData['vowel_ground_truth'] = metaData['vowel_ground_truth'] + vowel_ground_truth

        sentence_length_sequence = self.getSequence(metaData['encode_data'])

        padded_char = self.pad_test(metaData)

        for k,v in padded_char.items():
            padded_char[k] = v.T
        mask = self.mask(padded_char)

        if is_train:
            self._step = self._step + 1

        sentencePadding, sentencePadding_reverse, selectIndex = self.padSentence(ci_length_sequence)
        padded_sentence = {'forward':sentencePadding, 'backward':sentencePadding_reverse, 'valid_index':selectIndex }
        length_sequence = {'ci': ci_length_sequence,'sentence': sentence_length_sequence}

        return padded_char, padded_sentence, mask, length_sequence, yunjiao_list, yunjiao_gt_list
    # key next function to generate next batch
    def next_generation(self, is_train:bool=True) -> Tuple[Dict[str, List], Dict[str, List], List]:
        r"""next

        inputs: is_train
            - **is_train**: boolean flag to decide to activate train

        outputs: data_list, length_list, vocal_list

        notes:
            - same return type like evaluation, but affected by user instead random draw

        """

        metaData = {}
        metaData['encode_data'], metaData['decode_data'], metaData['result_data'],\
        metaData['tone'], metaData['vowel'] = [],[],[],[],[]
        ci_length_sequence = []
        yunjiao_list = []

        batchSize = self._batchSize if is_train else self._validateSize

        # FIXME should have better way than this loop and collection
        for i in range(batchSize):
            # key, index = self.drawKey(is_train)
            key, index = self.iterativeKey()
            encodeData = self.padEncode(self._data[key][index])
            decodeData = self.padDecode(self._data[key][index])
            resultData = self.padResult(self._data[key][index])
            tone = self.padTone(key,index)
            vowel,vowel_ground_truth,yunjiao,yunjiao_ground_truth = self.padVowel_test(key, index)
            # vowel, yunjiao = self.padVowel(key, index)

            yunjiao_list.append(yunjiao)
            ci_length_sequence.append(len(encodeData))
            metaData['encode_data'] = metaData['encode_data'] + encodeData
            metaData['decode_data'] = metaData['decode_data'] + decodeData
            metaData['result_data'] = metaData['result_data'] + resultData
            metaData['tone'] = metaData['tone'] + tone
            metaData['vowel'] = metaData['vowel'] + vowel

        sentence_length_sequence = self.getSequence(metaData['encode_data'])

        padded_char = self.pad(metaData)

        for k,v in padded_char.items():
            if k == 'vowel':
                padded_char[k] = tran(v)
            else:
                padded_char[k] = v.T
        mask = self.mask(padded_char)

        if is_train:
            self._step = self._step + 1

        sentencePadding, sentencePadding_reverse, selectIndex = self.padSentence(ci_length_sequence)
        padded_sentence = {'forward':sentencePadding, 'backward':sentencePadding_reverse, 'valid_index':selectIndex }
        length_sequence = {'ci': ci_length_sequence,'sentence': sentence_length_sequence}

        return padded_char, padded_sentence, mask, length_sequence, yunjiao_list
        # key next function to generate next batch
    def next_diversity(self, key,is_train:bool=True,is_semantic:bool=True):
        r"""next

        inputs: is_train
            - **is_train**: boolean flag to decide to activate train

        outputs: data_list, length_list, vocal_list

        notes:
            - same return type like evaluation, but affected by user instead random draw

        """

        metaData = {}
        metaData['encode_data'], metaData['decode_data'], metaData['result_data'],\
        metaData['tone'], metaData['vowel'], metaData['vowel_ground_truth'] = [],[],[],[],[],[]
        ci_length_sequence = []
        yunjiao_list = []
        yunjiao_gt_list= []

        batchSize = self._batchSize if is_train else self._validateSize

        # FIXME should have better way than this loop and collection
        for i in range(batchSize):
            # key, index = self.drawKey(is_train)
            key, index = self.fix_key(key,is_semantic)
            # key, index = self.draw_key_from_piror(is_semantic)
            encodeData = self.padEncode(self._data[key][index])
            decodeData = self.padDecode(self._data[key][index])
            resultData = self.padResult(self._data[key][index])
            tone = self.padTone(key,index)
            vowel,vowel_ground_truth,yunjiao,yunjiao_ground_truth = self.padVowel_test(key, index)
            # vowel, yunjiao = self.padVowel(key, index)
            yunjiao_gt_list.append(yunjiao_ground_truth)
            yunjiao_list.append(yunjiao)
            ci_length_sequence.append(len(encodeData))
            metaData['encode_data'] = metaData['encode_data'] + encodeData
            metaData['decode_data'] = metaData['decode_data'] + decodeData
            metaData['result_data'] = metaData['result_data'] + resultData
            metaData['tone'] = metaData['tone'] + tone
            metaData['vowel'] = metaData['vowel'] + vowel
            metaData['vowel_ground_truth'] = metaData['vowel_ground_truth'] + vowel_ground_truth

        sentence_length_sequence = self.getSequence(metaData['encode_data'])

        padded_char = self.pad_test(metaData)
        length = len(max(metaData['decode_data'], key = len))

        for k,v in padded_char.items():
            if k == 'vowel':
                padded_char[k] = tran(v)
            else:
                padded_char[k] = v.T
        mask = self.mask(padded_char)

        if is_train:
            self._step = self._step + 1

        sentencePadding, sentencePadding_reverse, selectIndex = self.padSentence(ci_length_sequence)
        padded_sentence = {'forward':sentencePadding, 'backward':sentencePadding_reverse, 'valid_index':selectIndex }
        length_sequence = {'ci': ci_length_sequence,'sentence': sentence_length_sequence}

        return padded_char, padded_sentence, mask, length_sequence, yunjiao_list, yunjiao_gt_list
    def eval_helper(self,key,index):
        metaData = {}
        metaData['encode_data'], metaData['decode_data'], metaData['result_data'],\
        metaData['tone'], metaData['vowel'], metaData['vowel_ground_truth'] = [],[],[],[],[],[]
        ci_length_sequence = []
        yunjiao_list = []
        yunjiao_gt_list= []

        


        encodeData = self.padEncode(self._data[key][index])
        decodeData = self.padDecode(self._data[key][index])
        resultData = self.padResult(self._data[key][index])
        tone = self.padTone(key,index)
        vowel,vowel_ground_truth,yunjiao,yunjiao_ground_truth = self.padVowel_test(key, index)
        yunjiao_gt_list.append(yunjiao_ground_truth)
        yunjiao_list.append(yunjiao)
        ci_length_sequence.append(len(encodeData))
        metaData['encode_data'] = metaData['encode_data'] + encodeData
        metaData['decode_data'] = metaData['decode_data'] + decodeData
        metaData['result_data'] = metaData['result_data'] + resultData
        metaData['tone'] = metaData['tone'] + tone
        metaData['vowel'] = metaData['vowel'] + vowel
        metaData['vowel_ground_truth'] = metaData['vowel_ground_truth'] + vowel_ground_truth

        sentence_length_sequence = self.getSequence(metaData['encode_data'])

        padded_char = self.pad_test(metaData)

        for k,v in padded_char.items():
            padded_char[k] = v.T
        mask = self.mask(padded_char)

        sentencePadding, sentencePadding_reverse, selectIndex = self.padSentence(ci_length_sequence)
        padded_sentence = {'forward':sentencePadding, 'backward':sentencePadding_reverse, 'valid_index':selectIndex }
        length_sequence = {'ci': ci_length_sequence,'sentence': sentence_length_sequence}

        return padded_char, padded_sentence, mask, length_sequence, yunjiao_list, yunjiao_gt_list
    def eval_next(self):
        correct_p_index, correct_t_index = self.get_user_intent()
        
        key, index = correct_p_index

        c = '语义: '
        print(c+key+'\n')

        print('*******************************')
        for x in self._cir_chars[key][index]:
            print(''.join(x))
        print('*******************************')

        padded_char, padded_sentence, mask, length_sequence, yunjiao_list, yunjiao_gt_list = self.eval_helper(key, index)
        key, index = correct_t_index
        padded_char_r, padded_sentence_r, mask_r, length_sequence_r, yunjiao_list_r, yunjiao_gt_list_r = self.eval_helper(key, index)
        return padded_char, padded_sentence, mask,length_sequence, yunjiao_list, yunjiao_gt_list,padded_char_r, padded_sentence_r, mask_r, length_sequence_r, yunjiao_list_r, yunjiao_gt_list_r
    def next_test_semantic_rule(self,keys,is_train=True):
        r"""next

        inputs: is_train
            - **is_train**: boolean flag to decide to activate train

        outputs: data_list, length_list, vocal_list

        notes:
            - same return type like evaluation, but affected by user instead random draw

        """
        # print('wwwww',keys)
        
        padded_char, padded_sentence, mask, length_sequence, yunjiao_list, yunjiao_gt_list = self.next_diversity(keys[0])
        padded_char_r, padded_sentence_r, mask_r, length_sequence_r, yunjiao_list_r, yunjiao_gt_list_r = self.next_diversity(keys[1],is_semantic=False)

        # metaData = {}
        # metaData['encode_data'], metaData['decode_data'], metaData['result_data'],\
        # metaData['tone'], metaData['vowel'], metaData['vowel_ground_truth'] = [],[],[],[],[],[]
        # ci_length_sequence = []
        # ci_length_sequence_semantic = []
        # yunjiao_list = []
        # yunjiao_gt_list= []

        # batchSize = self._batchSize if is_train else self._validateSize

        # # FIXME should have better way than this loop and collection
        # for i in range(batchSize):
        #     # key, index = self.drawKey(is_train)
        #     key_semantic, index_semantic = self.draw_key_from_piror()
        #     key_rule, index_rule = self.draw_key_from_piror(False)

        #     encodeData = self.padEncode(self._data[key_semantic][index_semantic])
        #     decodeData = self.padDecode(self._data[key_rule][index_rule])
        #     resultData = self.padResult(self._data[key_rule][index_rule])
        #     tone = self.padTone(self._tone[key_rule][index_rule])
        #     vowel,vowel_ground_truth,yunjiao,yunjiao_ground_truth = self.padVowel_test(key_rule, index_rule)
        #     yunjiao_gt_list.append(yunjiao_ground_truth)
        #     yunjiao_list.append(yunjiao)
        #     ci_length_sequence.append(len(decodeData))
        #     ci_length_sequence_semantic.append(len(encodeData))
        #     metaData['encode_data'] = metaData['encode_data'] + encodeData
        #     metaData['decode_data'] = metaData['decode_data'] + decodeData
        #     metaData['result_data'] = metaData['result_data'] + resultData
        #     metaData['tone'] = metaData['tone'] + tone
        #     metaData['vowel'] = metaData['vowel'] + vowel
        #     metaData['vowel_ground_truth'] = metaData['vowel_ground_truth'] + vowel_ground_truth

        # sentence_length_sequence = self.getSequence(metaData['decode_data'])
        # sentence_length_sequence_s = self.getSequence(metaData['encode_data'])

        # padded_char = self.pad_test_semantic_rule(metaData)

        # for k,v in padded_char.items():
        #     padded_char[k] = v.T
        # mask = self.mask(padded_char)

        # if is_train:
        #     self._step = self._step + 1

        # sentencePadding, sentencePadding_reverse, selectIndex = self.padSentence(ci_length_sequence)
        # sentencePadding_s, sentencePadding_reverse_s, selectIndex_s = self.padSentence(ci_length_sequence_semantic)
        # padded_sentence_s = {'forward':sentencePadding_s, 'backward':sentencePadding_reverse_s, 'valid_index':selectIndex_s }
        # padded_sentence = {'forward':sentencePadding, 'backward':sentencePadding_reverse, 'valid_index':selectIndex }
        # length_sequence_s = {'ci': ci_length_sequence_semantic,'sentence': sentence_length_sequence_s}
        # length_sequence = {'ci': ci_length_sequence,'sentence': sentence_length_sequence}

        return padded_char, padded_sentence, mask,length_sequence, yunjiao_list, yunjiao_gt_list,padded_char_r, padded_sentence_r, mask_r, length_sequence_r, yunjiao_list_r, yunjiao_gt_list_r
    # function to get user intent and return most relevant data
    def get_user_intent(self) -> Tuple[int,int]:
        r""" get user intent

        inputs: None

        outputs: tune_index, data_index
            -**tune_index**: an index of tune
            -**data_index**: an index of data given tune index

        notes:
           - user have to use chinses for keyword
        """

        data_set = self._data
        user_input = input("Enter key words:\n")
        # stupid loop to ensure user to type in the right thing
        while(not user_input.replace(" ", "").isalpha()):
            print("Key word must be all letters")
            user_input = input("Enter key words:\n")
        user_input = user_input.replace(" ", "")

        # use reverse dict to convert user input to ids
        r_dict = userHelper.reverse(self._vocab)
        user_input = userHelper.convert_to_id(user_input, r_dict)
        tune_lib = ioHelper.read(os.getcwd() + "/Data/title_lib.txt")

        # get key from the user
        tune_input = input("Enter tune number (1 ~ 818):\n")
        while(not tune_input.isdigit() or int(tune_input) < 1 or int(tune_input) > 818):
            print("Please enter valid number")
            tune_input = input("Enter tune number (1 ~ 818):\n")
        tune_input = int(tune_input) - 1

        # report the key in string format
        print("You selected:" + tune_lib[tune_input])
        tune_name = tune_lib[tune_input]

        # rank compare the each data with the entry from user
        # for a similarity score
        rank = []
        name = []
        for key in data_set:
            count = 0
            for i in data_set[key]:
                temp = userHelper.flatten(i)
                score = 0
                for j in user_input:
                    score += temp.count(int(j))
                rank.append(score)
                name.append([key, count])
                count += 1

        # get top result
        result = userHelper.max_n(np.asarray(rank), len(name))

        # use the top one as most match (p) , use the fisrt key match one as correct match (t)
        for i in reversed(result):
            if name[i[1]][0] == tune_name:
                correct_t_index = name[i[1]]
                break
        correct_p_index = name[result[-1][1]]

        return correct_p_index, correct_t_index


    # bunch of set, get function that no one loves
    def getData(self) -> List[List[List[int]]]:
        return self._data


    def getSequence(self, input:int) -> List[int]:
        # we need to delete 1 which is the adding eos from padEnc
        return [len(i)-1 for i in input]


    def getStep(self) -> int:
        return self._step


    def getEpoch(self) -> int:
        return self._epoch


    def setStep(self, step:int) -> None:
        self._step = step


    def setEpoch(self, epoch:int) -> None:
       self._epoch = epoch



# a = Iterator(config.get_config()[0])
# a.evaluation()
