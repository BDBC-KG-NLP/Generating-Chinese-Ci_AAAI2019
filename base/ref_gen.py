data = []
with open('Shixuehanying.txt',encoding='utf-8') as f:
    lines = f.readlines()
    classes = []
    for l in lines:
        if(l.startswith('<begin>')):
            continue
        if(l.startswith('<end>') ):
            data.append(classes)
            classes = []
            continue
        tmp = l.strip().split('\t')[1:]
        tmp = list(filter(lambda x: len(x)<3,[tmp[0]] + tmp[1].split(' ')))
        print(tmp)
        classes = classes + tmp

with open('~/ci_generation/base/Data/ci_per_line_after_cut.txt',encoding='utf-8') as f:
        lines = f.readlines()
        ci = []
        for l in lines:
            ci.append(l.strip().split('  '))


def check_word(word):
    for index,classes in enumerate(data):
        if word in classes:
            return index
    return -1


refer_dataset = []
for i_p,poetry in enumerate(ci):
    for i_s,s in enumerate(poetry):
        sen = []
        s = s.strip().split(' ')
        #print(s)
       # if(len(s)<4):
       #     continue
        for word in s:
            i = check_word(word)
            if(i>=0):
                sen.append(i)
        if(len(sen)>=2):
            print(sen)
            refer_dataset.append((i_p,i_s,sen))



