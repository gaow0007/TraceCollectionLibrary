import re 

p = re.compile('[0-9a-zA-Z_]+_abc')
print(p.match('124_abc'))