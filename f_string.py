
first_name = 'Corey'
last_name = 'Schafer'

sentence = 'My name is {} {}'.format(first_name, last_name)
print(sentence)

sentence = f'My name is {first_name.upper()} {last_name.upper()}'
print(sentence)

person = {'name': 'Jenn', 'age': 23}
# change open and close quotes with double quote
sentence = f"my name is {person['name']} and I am {person['age']} years old"
sentence

# zero padding
calculation = f'4 times 11 is equal to {4*11:04}' 
calculation

# Floating point value
pi = 3.1415926
sentence = f'Pi is equal to {pi:.4f}'
sentence

# Formatting Date
from datetime import datetime
bday = datetime(1993,10,11)
sentence = f'My bday is on {bday:%Y%m%d}'
sentence