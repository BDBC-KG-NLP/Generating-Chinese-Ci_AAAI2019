import numpy
import re
texts = []

with open('Data/mrcg.txt','r') as f:
    poems = f.read().split('\n\n')
    for poem in poems:
	    vectors = poem.split('!')
	    text = vectors.pop()
	    texts.append(text)
	 #   print(vectors)
	   
with open('Data/output.txt','w') as f:
    for text in texts:
       for t in text.split(' '):
           f.write(t)
           f.write('\n')
       f.write('\n\n')
      

