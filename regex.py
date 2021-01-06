import re
import pandas as pd

text_to_search = '''
abcdefghijklmnopqurtuvwxyz
ABCDEFGHIJKLMNOPQRSTUVWXYZ
1234567890
Ha HaHa
MetaCharacters (Need to be escaped):
. ^ $ * + ? { } [ ] \ | ( )
coreyms.com
321-555-4321
123.555.1234
123*555*1234
800-555-1234
900-555-1234
Mr. Schafer
Mr Smith
Ms Davis
Mrs. Robinson
Mr. T
'''
sentence = 'Start a sentence and then bring it to an end'

pattern = re.compile(r'(Mr|Ms|Mrs)\.? [A-Z][a-z]*')
m = pattern.search(text_to_search)
print(m)

urls = '''
https://www.google.com
http://coreyms.com
https://youtube.com
https://www.nasa.gov
'''

pattern = re.compile(r'https?://(www\.)?(\w+)(\.\w+)')

matches = pattern.finditer(urls)
for match in matches:
    print(match.group(2))
    

subbed_urls = pattern.sub(r'\2\3', urls)
print(subbed_urls)

matches = pattern.findall(urls)
for match in matches:
    print(match)

pattern = re.compile(r'start', re.I)
matches = pattern.search(sentence)
print(matches)

arr = ['a1', 'ab3', 'hello2', '3a4g']
df = pd.DataFrame(arr)
df

pattern = re.compile(r'[0-9]')

df['num'] = df[0].apply(lambda x: ''.join(pattern.findall(x)))
df['num'] = df[0].map(lambda x: ''.join(pattern.findall(x)))