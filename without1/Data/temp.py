import json
import sys

def dump(d, file):
    with open(file, 'w') as myfile:
        json.dump(d, myfile)


def load(file):
    with open(file, 'r') as myfile:
        d = json.load(myfile)
    return d

data = load(sys.argv[-1])
print(len(data),type(data),len(data['看花回'][1]),[len(x) for x in data['看花回'][1]])

print([len(xx) for x in data['看花回'] for xx in x])

print([[len(xx) for xx in x]for x in data['看花回'] ])