# -*- coding:utf-8 -*-

from nltk.translate.bleu_score import sentence_bleu
import codecs

class Bleu:
	def __init__(self,path):
		with codecs.open(path,'r',encoding='utf-8') as f:
			self.data = f.readlines()

		self.generation = []
		self.reference = []

		for i in range(len(self.data)-1):

			if (i-1)%3 == 0 :
				sentece = [word for word in self.data[i].strip() if word != '。' and word != '，' and word != ' ']
				self.generation.append(sentece)
			if i %3 == 0 :
				sentece = [word for word in self.data[i].strip() if word != '。' and word != '，' and word != ' ']
				self.reference.append(sentece)

	def score(self):
		final_score = 0
		for i in range(len(self.reference)):
			reference = []
			reference.append(self.reference[i])
			candidate = self.generation[i]
			score = sentence_bleu(reference, candidate,weights=(1,0,0,0))
			final_score += score
			print(reference)
			print(candidate)
			print('!!!!\n')
			if i > 2:
				break
			print(score)
		final_score = final_score/(i+1)
		return final_score

Bleu_score = Bleu('mrcg_abs.txt')
score = Bleu_score.score()

print('\n\n!!!!!!!!!!!!!')
print(score)
