import ioHelper
import jieba
import numpy as np


def write_ci_per_line():
	punct = set(u''':!),.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒
	﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠
	々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻
	︽︿﹁﹃﹙﹛﹝（｛“‘-—_…''')
	
	cir_chars = ioHelper.load('./Data/sim.json')
	f = open('./Data/ci_per_line.txt','w')
	for ci_pai_ming,cis in cir_chars.items():
		for i, each_ci in enumerate(cis):
			
			ci_str=''
			for each_sentence in each_ci:
				for char in each_sentence:
					if char in punct:
						ci_str+=' '
					else:
						ci_str+= char
			f.write(ci_str.strip()+'\n')
			print(ci_str)




def cut_all():
	d = {}
	f = open('./Data/ci_per_line.txt','r')
	fc = open('./Data/ci_per_line_after_cut.txt','w')
	for i,line in enumerate(f):

		seg_list = jieba.cut(line.strip(), cut_all=True)
		str_cut = ''
		for x in seg_list:
			str_cut += x+' '
			if x not in d:
				d[x] =1
			else:
				d[x] +=1
		fc.write(str_cut.strip() + '\n')
		if (i+1) % 100 ==0:
			print(i)
		# print(str_cut)
	ioHelper.dump(d,'./Data/cut_all.json')
	print(len(d))

def graph():
	# d = {}
	# f = open('./Data/ci_per_line.txt','r')
	# i = 0
	# for line in f:
	# 	for char in line:
	# 		if char != ' ':
	# 			if char not in d:
	# 				d[char] = i
	# 				i += 1
	# print(len(d))
	vocab_dim = 5949

	graph = np.zeros((vocab_dim,vocab_dim),dtype=np.int8)
	print(graph.shape,type(graph[0,0]))		
				
graph()
# cut_all()
# write_ci_per_line()
# sum1=0
# fenci = ioHelper.load('./Data/cut_all.json')
# for k,v in fenci.items():
# 	print(k,v)
# 	sum1+=v

# print(sum1,sum1/len(fenci))