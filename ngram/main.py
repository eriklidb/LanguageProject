import os
from ngram import NGramModel


def main():
    model_path = 'model.txt'
    if os.path.isfile(model_path):
        model = NGramModel.load(model_path)
    else:
        model = NGramModel(3)
    with open('vocab.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            pair = line.split(',')
            word = pair[0]
            if len(pair) >= 2:
                freq = int(pair[1])
            else:
                freq = 1
            model.learn(word)

    while True:
        print('> ', end='')
        word = input()
        if word == 'quit()' or word == '.q':
            break
        else:
            #model.learn(word)
            w = model.completions('the big boat was the', word, -1)
            print(w)

    model.save(model_path)

def context_and_keystrokes(text):
    if text.endswith(' '):
        pass

if __name__ == '__main__':
    main()
