import scipy.stats as stats



def coeff(path):
  print(path)
  x_list = []
  y_list = []
  with open(path,'r') as f:
     
     for line in f.readlines()[:-1]:
       if 'minus' in path:
          item  = line.split(' ')
       else:
          item = line.split('  ')
       x_list.append(float(item[0]))
       y_list.append(float(item[1]))
         
  print('coff:',stats.pearsonr(x_list,y_list)[0])


coeff('Data/score_mrcg_minus.txt')
coeff('Data/score_mrcg.txt')


