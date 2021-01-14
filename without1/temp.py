import statistics_hardcoded

# from config import get_config
# import iterator
# # print(cipai_cuml)

# config, unparsed = get_config()
# config.batch_size = 5
# step = iterator.Iterator(config)
# padded_char, padded_sentence, mask, length_sequence, yunjiao_list = step.next()

# print(yunjiao_list)
# print(length_sequence)
# cum = 0
# d = {}
# for i,yuns in enumerate(yunjiao_list,):
# 	if i>0:
# 		cum += length_sequence['ci'][i-1]
# 	for key,values in yuns.items():
# 		for v in values:
# 			d_key = length_sequence['sentence'][cum + v]-2
# 			if d_key in d:
# 				d[d_key].append((cum + v,key))
# 			else:
# 				d[d_key] = [(cum + v,key)]
# print(d)

print(3 in statistics_hardcoded.yunjiao_stat[6])