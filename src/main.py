from translate import translate_sentence

while True:
    
    en = input("Nhập câu tiếng Anh: ")
    if en == '0':
        break
    de_translation = translate_sentence(en)
    print(f'de: {de_translation}')