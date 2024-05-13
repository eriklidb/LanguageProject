import os
from ngram import NGramModel
from data import DataSource, clean, context_and_keystrokes

def main():
    data_path = 'data'
    model_path = 'model.txt'
    
    k = 3
    source = DataSource(data_path)
    source.save_samples('samples.txt', 0, k)
    if os.path.isfile(model_path):
        print('Loading...')
        model = NGramModel.load(model_path)
        print('Done loading.')
    else:
        print('Teaching...')
        model = NGramModel(k)
        for sample in source.labeled_samples(0, k-1):
            model.learn_sample(*sample)
        print('Done teaching.')
        print('Saving...')
        model.save(model_path)
        print('Done saving.')
    
    
    while True:
        print('(.q to quit) > ', end='')
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
