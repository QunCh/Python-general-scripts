
from collections import Counter
from collections import defaultdict

############### Counter
a = Counter('gallon')
b = Counter(['a', 'a', 'b', 'c', 'c'])
c = Counter({'o': 5, "p" : 3, 'd': 2})
d = Counter(cat = 8, dog = 2)
print(a)
print(b)
print(c)
print(d)

c['o']
list(c.elements())


c.most_common(2)

c.subtract('opppppp')
#Can use dict/list/Counter...
#c.update : adding
#c.clear() : clear

a-b  #return no 0/negative value
c+d
a & b   # intersaction
a | b   # union

###################### Defaultdict
adict = defaultdict(int)
adict = defaultdict(lambda: 0)

for x in ['a', 'a', 'b', 'c', 'c']:
    adict[x] = adict[x]+1
dict(adict)





