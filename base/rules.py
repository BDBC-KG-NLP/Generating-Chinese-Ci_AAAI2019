import operator
import collections
import random
import sys
import os
from ioHelper import *





rule_known_punctuation = '韵叶叠重'
# rule_known_punctuation = ['韵','叶','叠','重']
known_punctuation = [4, 541, 1705]
# 。 ？ ！

def check_recovery(ref,generated):
    counter = 0
    total = 0
    for xs, x_hats in zip(ref, generated):
        for x,x_hat in zip(xs[:-1],x_hats):
            # only check upto xs[:-1]
            if x == x_hat:
                counter += 1
            total += 1
    if counter == 0:
        return 0.
    return counter / total


def check_length(ref,generated,strict = False):
    # l = [len(x_hats) / len(xs) for xs, x_hats in zip(ref, generated)]
    counter = 0
   
    for xs, x_hats in zip(ref, generated):
        if len(x_hats) == len(xs):
            counter += 1
    if strict:
        if counter == len(generated):
            return 1
        else:
            return 0
    return counter / len(generated)

def check_pingze(ref, generated, vocab_lookup, tone_lookup):
    no_sound = 0

    counter = 0
    total = 0
    for xs, x_hats in zip(ref, generated):
        for x,x_hat in zip(xs,x_hats):

            if tone_lookup[vocab_lookup[x]] == no_sound:
                counter += 1
            elif tone_lookup[vocab_lookup[x]] == tone_lookup[vocab_lookup[x_hat]]:
                counter += 1
            total += 1
    if total == 0:
        return 0.0
    return counter / total

def check_yunshe(random_yunjiao, generated, vocab_lookup, vowel_lookup, ref=None):
    counter = 0
    total = 0

    # print(random_yunjiao)
    for key,value in random_yunjiao.items():
        temp ={}
        for i in value:

            # print(vowel_lookup[vocab_lookup[generated[i][-2]]],key,type(vowel_lookup[vocab_lookup[generated[i][-2]]]),type(key))
            # if len(generated[i]) > 1 and vowel_lookup[vocab_lookup[generated[i][-2]]] == key:
            #     counter += 1
            key = -1
            if len(generated[i])>1:
                if ref is None:
                    key = vowel_lookup[vocab_lookup[generated[i][-2]]]
                else:
                    pos = len(ref[i]) - 2
                    if len(generated[i]) > pos:
                        key = vowel_lookup[vocab_lookup[generated[i][pos]]]
                    # print('omg:',i,vocab_lookup[generated[i][pos]],key)
                    # if generated[i][pos] == 2:
                    #     # print('found you!!!!!')
                    #     sys.exit()
                if key != -1:
                    if key in temp:
                        temp[key] += 1
                    else:
                        temp[key] = 1
        wtf=sorted(temp, key=temp.get, reverse=True)
        if wtf:
            counter += temp[wtf[0]]
        total += len(value)
    if total == 0:
        return 0.0
    return counter / total


def find_potential_rules_by_title(rules, cir):
    """find the rules that has same number of lines and
        each line has same length of chars

    :param rules: all rules w.r.t the specific title
    :type rules: list of list of list
    :param cir: the reference of ci which is already encoded
    :type cir: list of list
    :return: matched_rules
    :rtype: list of list of list
    """

    compare = lambda x, y: collections.Counter(x) == collections.Counter(y)

    rule_length = [sum(len(x) for x in rule) for rule in rules]
    cir_length = sum(len(x) for x in cir)
    matched_rules = []
    for i, x in enumerate(rule_length):
        if cir_length == x:
            cir_sentences_length = [len(x) for x in cir]
            rule_sentences_length = [len(x) for x in rules[i]]
            # print('www',cir_sentences_length,rule_sentences_length)
            if compare(cir_sentences_length, rule_sentences_length):
                matched_rules.append(rules[i])
    return matched_rules


def pz_mapping(x):
    """mapping vocal character to id

    :param x: vocal character in rule
    :type x: char
    :return: id w.r.t to x
    :rtype: int
    """
    if x == '平':
        return 1
    elif x == '中':
        return 0
    elif x == '仄':
        return 2
    else:
        # print('wrong pingze mapping')
        return 0



def check_rules_based_on_pingze(cir_tone, matched_rules):
    """check the tone of reference ci w.r.t matched_rules

    :param cir_tone: the tone information for the referenced ci
    :type cir_tone: list of list
    :param matched_rules: the list of rules given the title,corresponding to
            number of lines and length of each lines
    :type matched_rules: list of list of list
    :return: the best matched rule and corresponding score
    :rtype: list of list,float
    """
    counter = 0
    total = 0
    score = {}

    for i, rule in enumerate(matched_rules):
        for ci_sentence, rule_sentence in zip(cir_tone, rule):
            for x, y in zip(ci_sentence[:-1], rule_sentence[:-1]):

                if pz_mapping(y) == 0 or int(x) == 0 or pz_mapping(y) == int(x):
                    counter += 1
                total += 1
        score[i] = counter / total

    score_sorted_list_of_tuple = \
        sorted(score.items(), key=operator.itemgetter(1), reverse=True)

    (i, score) = score_sorted_list_of_tuple[0]
    # print(matched_rules[i])
    # print('*************')
    # matched_rules[i] = [[pz_mapping(char) for char in line ] for line in matched_rules[i]]
    
    return matched_rules[i], score


def group_yunshe(rule, cir, cir_vowel):
    """get yunjiao by group, rule can be specified or not, if not, we assume
        the position of yunjiao is before any in rule_known_punctuation

    :param rule:the best matched rule or not given
    :type rule: list of list or empty list
    :param cir: the referenced ci which is already encoded
    :type cir: list of list
    :param cir_vowel: the encoded vowel info for referenced ci
    :type cir_vowel: list of list
    :return: group yunshe with corresponding line index
    :rtype: dict
    """
    yunshe = {}
    if rule is not None:
        for i, ci_sentence in enumerate(rule):
            yunjiao = ci_sentence[-2:]
            if yunjiao[1] in rule_known_punctuation:
                if cir_vowel[i][-2] in yunshe:
                    yunshe[cir_vowel[i][-2]].append(i)
                else:
                    yunshe[cir_vowel[i][-2]] = [i]
    else:
        for i, ci_sentence in enumerate(cir):
            yunjiao = ci_sentence[-2:]
            # print('yunjiao',yunjiao)
            if len(yunjiao)==1:
                print('sentence_length==1!:',yunjiao,len(ci_sentence),'txt',ci_sentence)
            elif yunjiao[1] in known_punctuation:
                if cir_vowel[i][-2] in yunshe:
                    yunshe[cir_vowel[i][-2]].append(i)
                else:
                    yunshe[cir_vowel[i][-2]] = [i]
    # print('original_yunjiao:', yunshe)
    return yunshe


def yunshe_statistics(vowel_dict):
    """cumulative interval based on vowel info from the whole corpus

    :param vowel_dict: vowel info for the whole corpus
    :type vowel_dict: dict of list of list of list
    :return: cumulative interval
    :rtype: list of int
    """
    yunshe_stat = {}
    no_sound = 16
    total_yunshe = 17
    for _, cis in vowel_dict.items():
        for ci in cis:
            for sentence in ci:
                for each_vowel in sentence:
                    if each_vowel != no_sound:
                        if each_vowel in yunshe_stat:
                            yunshe_stat[each_vowel] += 1
                        else:
                            yunshe_stat[each_vowel] = 1
    # for each,v in yunshe_stat.items():
    # 	print(each,v)
    cum = 0
    cuml = []
    for i in range(total_yunshe):
        if i != no_sound:
            cum += yunshe_stat[i]
            cuml.append(cum)
    return cuml


def random_yunshe(cuml, drawn):
    """ generate yunshe randomly

    :param cuml: cumulative interval
    :type cuml: list of int
    :param drawn: a int drawn form the whole range
    :type drawn: int
    :return: random_yunshe
    :rtype: int
    """
    for i, e in enumerate(cuml):
        if drawn < e:
            break
    return i


def random_yunjiao(yunjiao, cuml):
    """assign random yunshe to yunjiao position

    :param yunjiao: origin yunjiao info group by yunshe
    :type yunjiao: dict
    :param cuml: cumulative interval
    :type cuml: list of int
    :return: random_yunjiao after assign
    :rtype: dict
    """
    random_yunjiao_dict = {}
    for key, value in yunjiao.items():
        yunshe = random_yunshe(cuml, random.choice(range(cuml[-1])))
        while yunshe in random_yunjiao_dict:
            yunshe = random_yunshe(cuml, random.choice(range(cuml[-1])))
        random_yunjiao_dict[yunshe] = value
    # print('random_yunjiao:', random_yunjiao_dict)
    return random_yunjiao_dict


# def get_yunjiao_testing(rules, cir_ids, cir_tone, cir_vowel, cuml):
#     """ main method for generate vowel.
#             1)get same lengthe's rule
#             2)find best match rule
#             3)assign random yunshe to yunjiao

#     :param rules: all rules w.r.t the specific title
#     :type rules: list of list of list
#     :param cir_ids: encoded referenced ci
#     :type cir_ids:
#     :param cir_tone: encoded referenced ci's tone info
#     :type cir_tone:
#     :param cir_vowel: encoded referenced ci's vowel info
#     :type cir_vowel:
#     :param cuml: cumulative interval
#     :type cuml:
#     :return:
#     :rtype: new vowel infor after assign random yunshe
#     """
#     same_sentence_length_rules = \
#         find_potential_rules_by_title(rules, cir_ids)
#     if not same_sentence_length_rules:
#         original_yunjiao = group_yunshe([], cir_ids, cir_vowel)
#     else:
#         best_matched_rule, rule_best_matched_score = \
#             check_rules_based_on_pingze(cir_tone, same_sentence_length_rules)

#         # we may need a score threshold
#         # print('rule_best_matched_score:', rule_best_matched_score)
#         original_yunjiao = group_yunshe(best_matched_rule, cir_ids, cir_vowel)
#     random_yunjiao_dict = random_yunjiao(original_yunjiao, cuml)
#     return random_yunjiao_dict


def get_yunjiao_training(rules, cir_ids, cir_tone, cir_vowel):
    """same as above except for training
        current under development

    :param rules:
    :type rules:
    :param cir_ids:
    :type cir_ids:
    :param cir_tone:
    :type cir_tone:
    :param cir_vowel:
    :type cir_vowel:
    :return:
    :rtype:
    """

    same_sentence_length_rules = \
        find_potential_rules_by_title(rules, cir_ids)

    if not same_sentence_length_rules:
        # print('no rule founded!')
        original_yunjiao = group_yunshe(None, cir_ids, cir_vowel)
    else:
        best_matched_rule, rule_best_matched_score = \
            check_rules_based_on_pingze(cir_tone, same_sentence_length_rules)

        # we may need a score threshold
        # print('rule_best_matched_score:', rule_best_matched_score)
        original_yunjiao = group_yunshe(best_matched_rule, cir_ids, cir_vowel)
    return original_yunjiao
    # return None
    
def get_tone_by_rule(rules, cir_ids, cir_tone,get_tone_by_rule):

    if  not get_tone_by_rule:
        return cir_tone

    same_sentence_length_rules = \
        find_potential_rules_by_title(rules, cir_ids)

    if not same_sentence_length_rules:
        # print('no rule founded!')
        return cir_tone
    else:
        best_matched_rule, rule_best_matched_score = \
            check_rules_based_on_pingze(cir_tone, same_sentence_length_rules)
        if rule_best_matched_score > 0.9:
            best_matched_rule = [[pz_mapping(char) for char in line ] for line in best_matched_rule]
            return best_matched_rule
        else:
            return cir_tone


# def change_dict__str_to_int():
#     """ a helper function to change str type of element in tone_dict to int
#             should be depreciated and removed later

#     :return:
#     :rtype:
#     """
#     tone_dict = load('./data/shicis_tone_dict_str.json')
#     for key, cis in tone_dict.items():
#         for i, ci in enumerate(cis):
#             for j, sen in enumerate(ci):
#                 tone_dict[key][i][j] = [int(x) for x in sen]

#     dump(tone_dict, './shicis_tone_dict.json')


# cir_vowel=load(os.getcwd()+'/Data/vowel.json')
# cir_tone=load(os.getcwd()+'/Data/tone.json')
# cir_ids=load(os.getcwd()+'/Data/num.json')
# cir_chars=load(os.getcwd()+'/Data/sim.json')
# rules=load(os.getcwd()+'/Data/rules.json')
# # cuml=yunshe_statistics(cir_vowel)
# print(cir_chars['满江红'][8])
# # print(get_generate_vowel(rules['茶瓶儿'],cir_ids['茶瓶儿'][0],cir_tone['茶瓶儿'][0],cir_vowel['茶瓶儿'][0],cuml))

# matched_rules=find_potential_rules_by_title(rules['满江红'],cir_ids['满江红'][8])
# br,bs = check_rules_based_on_pingze(cir_tone['满江红'][8],matched_rules)
# # print(matched_rules)
# # print('************************')
# # print(rules['声声慢'])
# print(br,bs)
# print('************************')
# print(cir_tone['满江红'][8])
# rule,score=check_pingze(cir_tone['声声慢'][8],matched_rules)
# print(rule,score)
# print(group_yunshe(rule,cir_ids['声声慢'][8],cir_vowel['声声慢'][8]))

# rules=load('./data/rule.json')
# cc=0
# total=0
# for key, rs in rules.items():

#     for ci in rs:
#         for each_rule in rs:
            
#             for each_sentence in each_rule:
#                 total+=1
#                 for i,char in enumerate(each_sentence):
#                     if char == '叶':
#                         print(key,each_sentence)
#                     if char in rule_known_punctuation:
                        
#                         if i != len(each_sentence)-1:
#                             print(key,each_sentence)
#                             cc+=1
# print(cc,total)


# print(len(rules['声声慢']))
# print(rules['声声慢'])

# cuml=yunshe_statistics(vowel_dict)
# x= random.choice(range(cuml[-1]))
# print(x)
# print(cuml)
# for i,e in enumerate(cuml):
# 	if x<e:
# 		print(i)
# 		break


# def read_rule(path):
#
# cir
# cig
# rule
#
# # find rules corresspound to cipai title
# find_rules_by_title(rules,title,cir)
# #1 check pingze
