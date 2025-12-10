from torchtext.datasets import Multi30k
from translate import translate_sentence

import random


train = Multi30k(split='train', language_pair=('en', 'de'))
valid = Multi30k(split='valid', language_pair=('en', 'de'))
test = Multi30k(split='test', language_pair=('en', 'de'))


# print(f"Số câu trong tập test : {len(list(test))}")


train_list = list(train)
valid_list = list(valid)

print(f"Số câu trong tập train: {len(train_list)}")
print(f"Số câu trong tập valid: {len(valid_list)}")



n = random.randint(0, 29000)
en_sent, de_sent = train_list[n]
    
print(f"EN            : {en_sent}")
print(f"DE translation: {translate_sentence(en_sent)}")
print(f"DE            : {de_sent}")

    
