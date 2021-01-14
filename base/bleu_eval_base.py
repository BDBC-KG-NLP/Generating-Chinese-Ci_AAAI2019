import bleu
import jieba
import re
from nltk.translate.bleu_score import sentence_bleu,corpus_bleu
from random import sample
     

def get_reference():
    refer_dataset = []
    with open('Data/reference.txt','r',encoding='utf-8') as f:
        lines = f.readlines()
        for l in lines:
           ref = []
           reference = re.search('\[.*\]',l).group()
           index_p,index_s= l.strip().split(' ')[0:2]
           index_p = int(index_p)
           index_s = int(index_s)
           for i in reference:
             if(i.isdigit()):
                 ref.append(i)
           ref = [int(i) for i in ref]
           ref = sorted(ref)
           refer_dataset.append((index_p,index_s,ref))
    print('Reference Ready!\n')
    return refer_dataset

def get_output():
    output_dataset = []
    with open('Data/output.txt','r',encoding='utf-8') as f:
         flag = 1
         lines = f.readlines()
         for l in lines:
                 l = l.strip()
                 output_dataset.append([i for  i in jieba.cut(l)][:-1])
                 print(output_dataset)
   # output_dataset = output_dataset.remove([])
    print('Output Ready!\n')
    return output_dataset

def get_Shixue():
    data = []
    with open('Data/Shixuehanying.txt',encoding='utf-8') as f:
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
    #       print(tmp)
           classes = classes + tmp
    print('Shixuehanying Ready!\n')
    return  data

def get_ci():
   with open('Data/ci_per_line.txt',encoding='utf-8') as f:
        lines = f.readlines()
        ci = []
        for l in lines:
            ci.append(l.strip().split('  ')[0].split(' '))
   return ci

Shixue = get_Shixue()
Ref = get_reference()
Ci  = get_ci()



def check_word(word):
    for index,classes in enumerate(Shixue):
        if word in classes:
            return index
    return -1

def is_sub_set(list1, list2):
    flag = False
    for i in range(len(list2) - len(list1) + 1):
        if list2[i: i+len(list1)] == list1:
            flag = True
            break
    return flag

def get_next_ref(sen):
    sen_list = []
    refers = []
    for word in sen:
         i = check_word(word)
         if(i>=0):
             sen_list.append(i)
    sen_list = sorted(sen_list)
    if(sen_list!=[]):
    #   print(sen_list)
       for poten_ref in Ref:
       #    print(poten_ref[2])
           if(is_sub_set(sen_list,poten_ref[2])):
                 try:#如果poten_ref[1]+1 out of index, 就粗暴的continue
                      refers.append([i for i in Ci[poten_ref[0]][poten_ref[1]+1]])
                       
                 except:
                      continue
    return refers


def assign_ref(output_dataset):
    references = []
    outputs    = []
    for i in range(len(output_dataset)-1):
       if(output_dataset[i] == [] or output_dataset[i+1]==[]):
           continue
       else:
           if(len(output_dataset[i])<=3):
              continue
           refers = get_next_ref(output_dataset[i])
          # print(refers)
           if(len(refers)>=20):
              refers = sample(refers,20)
           elif(len(refers)<=3):
              continue
           output_dataset[i+1] = [i for i in ''.join(output_dataset[i+1])]
           bleu_score = sentence_bleu(refers,output_dataset[i+1],weights=(1,0,0,0))
           references.append(refers)
           outputs.append(output_dataset[i+1])
          # bleu_score, _, _, _, _, _ = bleu.compute_bleu(output_dataset[i+1], refers)
           print(output_dataset[i+1],'\t','Bleu:',bleu_score,)
          # print('References:\n',refers,'\n') 
           
    return references,outputs


            
output_dataset = get_output() 
references,outputs = assign_ref(output_dataset) 
print('Final Bleu Score:',corpus_bleu(references,outputs,weights=(1,0,0,0)))

