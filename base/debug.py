import ioHelper
import rules
import operator
import random
import statistics_hardcoded

def debug_yunshe_statistics():

    vowel = ioHelper.load("./Data/vowel.json")
    cuml = rules.yunshe_statistics(vowel)
    decuml =[]
    for i,x in enumerate(cuml):
        if i == 0:
            decuml.append(cuml[i])
        else:
            decuml.append(cuml[i] - cuml[i-1])

    print(len(cuml),sum(cuml))
    print(cuml)
    print([ _ / sum(cuml) for _ in cuml])
    print(sum([ _ / sum(cuml) for _ in cuml]))

    print('*********************************')
    print(len(decuml),sum(decuml))
    print(decuml)
    print([ _ / sum(decuml) for _ in decuml])
    print(sum([ _ / sum(decuml) for _ in decuml]))


# [0.004201425661918796, 0.02074971222171436, 0.02823308656344009, 0.03202931392758344, 0.04182057719746047, 0.0450857934057926, 0.0543549013826237, 0.060946239143931, 0.06387557121113141, 0.07475172024602013, 0.07810749616638003, 0.07922635778950086, 0.08922198712319235, 0.0978092788196225, 0.11140441189290426, 0.11818212724678401]

# 16 5551550
# [197360, 777348, 351528, 178326, 459940, 153382, 435412, 309625, 137604, 510902, 157636, 52558, 469540, 403384, 638625, 318380]

# [0.03555043186137205, 0.1400235970134467, 0.063320694220533, 0.032121839846529346, 0.08284893408147274, 0.027628680278480786, 0.07843070854085796, 0.05577271212544244, 0.024786591132206322, 0.09202871270185804, 0.028394952760940638, 0.009467265898712973, 0.08457818086840613, 0.07266150894795147, 0.11503544055263845, 0.057349749169150956]
def rules_stat():

    total_chars = 0
    total_sentences = 0
    char_level = {'平':0,'仄':0,'中': 0}
    sentence_level = {'韵':0,'叶':0,'叠':0,'重':0}

    rule_set = ioHelper.load("./Data/rules.json")
    for k,v in rule_set.items():
        for each_rule in v:
            for each_sentence in each_rule:
                for char in each_sentence:
                    if char in char_level:
                        char_level[char] +=1
                    elif char in sentence_level:
                        sentence_level[char] +=1

                    total_chars +=1
                total_sentences+=1

    print('char_level:',char_level,'total_chars',total_chars)
    for k,v in char_level.items():
        char_level[k] = v / total_chars
    print('char_level_stat:', char_level)
    print('sentence_level:', sentence_level, 'total_sentences', total_sentences)
    for k, v in sentence_level.items():
        sentence_level[k] = v / total_sentences
    print('sentence_level_stat:', sentence_level)

# char_level: {'中': 10218, '平': 86832, '仄': 85581} total_chars 218272
# char_level_stat: {'中': 0.046813150564433366, '平': 0.39781556956458, '仄': 0.3920841885354054}
# sentence_level: {'叶': 285, '韵': 19850, '叠': 142, '重': 24} total_sentences 35641
# sentence_level_stat: {'叶': 0.007996408630509806, '韵': 0.5569428467214724, '叠': 0.003984175528183833, '重': 0.0006733817794113521}

def eval_rules_stat():
    rule_set = ioHelper.load("./Data/rules.json")
    cir_chars = ioHelper.load('./Data/sim.json')
    tone = ioHelper.load("./Data/tone.json")
    no_rule_matched_count = 0
    matched_count = 0
    scores = {}
    ci_pai_count = {}
    for ci_pai_ming,cis in cir_chars.items():
        ci_pai_count[ci_pai_ming] = len(cis)
        for i, each_ci in enumerate(cis):
            same_sentence_length_rules = rules.find_potential_rules_by_title(rule_set[ci_pai_ming],each_ci)
            if not same_sentence_length_rules:
                no_rule_matched_count += 1
            else:
                _, rule_best_matched_score = rules.check_rules_based_on_pingze(
                    tone[ci_pai_ming][i], same_sentence_length_rules)
                if ci_pai_ming in scores:
                    scores[ci_pai_ming].append(rule_best_matched_score)
                else:
                    scores[ci_pai_ming] = [rule_best_matched_score]
                matched_count += 1
    print('total ci:',matched_count+no_rule_matched_count, 'rule_matched_ci',matched_count,
          'no_rule_matched',no_rule_matched_count)
    print('\n')
    print('group by ciapi count',ci_pai_count)
    print('\n')
    print('rule_matched_ci percentage:',matched_count/(matched_count+no_rule_matched_count))
    print('\n')
    print('no_rule_matched_ci percentage:', no_rule_matched_count / (matched_count + no_rule_matched_count))
    print('\n')
    print('group by matched cipai count:',[k+' : '+str(len(v)) for k,v in scores.items()])
    print('\n')
    print('group by matched cipai stat:',[k+' : '+str(len(v)/ci_pai_count[k]) for k, v in scores.items()])
    print('\n')
    print('group by matched cipai score avg:', [k+' : '+str(sum(v)/len(v)) for k, v in scores.items()])
    print('\n')
    flatten = [vv for k, v in scores.items() for vv in v]
    print('total score avg:',sum(flatten)/len(flatten))

def cipai_stat():
    cir_chars = ioHelper.load('./Data/sim.json')

    ci_pai_count = {}
    for ci_pai_ming,cis in cir_chars.items():
        ci_pai_count[ci_pai_ming] = len(cis)
    sorted_x = sorted(ci_pai_count.items(), key=operator.itemgetter(1), reverse=True)
    print(sorted_x)
    # cuml = 0
    # cumlist = []
    # for x in sorted_x:
    #     cuml += x[1]
    #     cumlist.append((x[0],cuml))

    # print(cumlist,len(cumlist))
    # print(ci_pai_count,ci_pai_count['浣溪沙'])
    # print(cumlist['抛毬乐'])

def yunjiao_stat():
    d ={}
    vowel = ioHelper.load("./Data/vowel.json")
    data = ioHelper.load("./Data/num.json")
    for ci_pai_ming,vow in vowel.items():
        for i, each_ci_vow in enumerate(vow):
            for j,each_sentence in enumerate(each_ci_vow):
                if len(each_sentence)>2:
                    if each_sentence[-2] in d:
                        d[each_sentence[-2]].append(data[ci_pai_ming][i][j][-2])
                    else:
                        d[each_sentence[-2]] = [data[ci_pai_ming][i][j][-2]]
    
    for dd in d:
        d[dd] = list(set(d[dd]))
    print(d)          
    return d

def debug_cipai_dist():
    sample_times = 10000
    n = statistics_hardcoded.cipai_cuml[-1][1]
    d={}
    for i in range(sample_times):
        draw = random.choice(range(n))
        for j,x in enumerate(statistics_hardcoded.cipai_cuml):
            if draw < x[1]:
                key = x[0]
                if key not in d:
                    d[key] = 1
                else:
                    d[key] +=1
                break

    sorted_sample_x = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
    sorted_sample_x_prob = [(i,j/sample_times) for (i,j) in sorted_sample_x]

    cir_chars = ioHelper.load('./Data/sim.json')
    total = 0
    ci_pai_count = {}
    for ci_pai_ming,cis in cir_chars.items():
        ci_pai_count[ci_pai_ming] = len(cis)
        total += len(cis)
    sorted_x = sorted(ci_pai_count.items(), key=operator.itemgetter(1), reverse=True)
    sorted_x_prob = [(i,j/total) for (i,j) in sorted_x]

    l1=0
    l2=0
    print('emperical freq / sample freq / l1 / l2')
    for x,x_sample in zip(sorted_x_prob,sorted_sample_x_prob):
        print(x[0],x[1],x_sample[1],abs(x[1]-x_sample[1]),(x[1] - x_sample[1])**2)
        l1 += abs(x[1]-x_sample[1])
        l2 += (x[1] - x_sample[1])**2

    print(l1/len(sorted_x_prob), l2/len(sorted_x_prob))

debug_yunshe_statistics()
# rules_stat()
# eval_rules_stat()
# cipai_stat()
#debug_cipai_dist()
# yunjiao_stat()
