import ioHelper
import os

data = ioHelper.load(os.getcwd()+"/Data/num.json")
key_list = list(data.keys())
for k in key_list:
   print(len(data))
   break
