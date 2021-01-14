from nltk.translate.bleu_score import  sentence_bleu
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
poems = []
scores = []

with open('Data/mrcg_minus.txt','r') as f:
    temp_list = f.read().split('\n\n')[:-1]
    for temp in temp_list:
        info = temp.split('\n')
        vector = np.array([float(i) for i in  info[0].split(' ') if i is not ''])
        text = [i for i in info[1] if i not in '，,。!？?. ']
        poems.append((vector,text))
poems = poems[:1000]

class Pair():
   def __init__(self,p1,p2,sim,bleu):
      self.p1 = p1
      self.p2 = p2
      self.sim = sim
      self.bleu = bleu



def cos_sim(vector1,vector2):
    return np.dot(vector1,vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2)))


def compare_pair(p1,p2):
    sim = cos_sim(p1[0],p2[0])
    bleu = sentence_bleu([p1[1]],p2[1],weights=(1,0,0,0))
    P  =  Pair(p1,p2,sim,bleu)
    return P

def compare_all():
    for i in range(len(poems)-1):
        for j in range(i+1,len(poems)):
           item = compare_pair(poems[i],poems[j])
           with open('Data/score_mrcg_minus.txt','a+') as f:
                f.write(str(item.sim)) 
                f.write(' ')
                f.write(str(item.bleu))
                f.write('\n')
     
compare_all()
#with open('Data/score_mrcg.txt','w') as f:
#    for item in scores:
#        f.write(item.sim,' ',item.bleu,'\n' )

           
