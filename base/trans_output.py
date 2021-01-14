with open('Eval_Data/Guided/mrcg_abs.txt','r') as f:
    lines = f.readlines()
    with open('Test/mrcg_test.txt','w') as fp:
        for i,l in enumerate(lines):
            if(i % 3 == 1):
                lines = l.strip().split(' ')
                print(lines)
                for line in lines:
                    fp.write(line)
                    fp.write('\n')
                fp.write('\n\n')


          
