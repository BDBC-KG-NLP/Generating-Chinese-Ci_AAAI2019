import sys
def format_input(temp):
	max_len = max(len(_) for _ in temp)
	real_max = (max_len)//5
	while temp:
		x1 = temp.pop(0)
		print(temp)
		len_x1= len(x1)
		real_x1 =(len_x1)//5
		
		x2 = temp.pop(0)
		diff = max_len-len_x1 + real_max-real_x1
		print(x1+' '*diff+' | '+x2)

f = open(sys.argv[1],'r')

temp = []
for line in f:
	if line.startswith('====>'):
		temp.append(line[10:-1])
	else:
		if len(temp)>0:
			format_input(temp)
			print(line)
		else:
			print(line)
print(sys.argv[1])