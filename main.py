import os
from ngram import NGramModel
from data import DataSource
from common import context_and_keystrokes

def main():
    data_path = 'data'
    model_path = 'model.txt'
    
    source = DataSource(data_path)
    if os.path.isfile(model_path):
        print('Loading...')
        model = NGramModel.load(model_path)
        print('Done loading.')
    else:
        print('Teaching...')
        model = NGramModel(3)
        for sentence in source.sentences():
            model.learn(sentence)
        print('Done teaching.')
        print('Saving...')
        model.save(model_path)
        print('Done saving.')
    

    while True:
        print('> ', end='')
        text = input()
        if text == 'quit()' or text == '.q':
            break
        else:
            context, keystrokes = context_and_keystrokes(text)
            completions, probs = model.completions(context, keystrokes, 3)
            print(completions)
            print(probs)
    

if __name__ == '__main__':
    main()
