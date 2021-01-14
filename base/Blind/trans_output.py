import codecs
# 对于StreamReader和StreamWriter的简化， codecs模块提供一个open方法
with codecs.open('../test/att_test.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
    with codecs.open('Data/output.txt','w',encoding='utf-8') as fp:
        for i,l in enumerate(lines):
            if(i % 3 == 1):
                lines = l.strip().split(' ')
                print(lines)
                for line in lines:
                    fp.write(line)
                    fp.write('\n')
                fp.write('\n\n')


          
