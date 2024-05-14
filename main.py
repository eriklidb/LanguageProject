# Authors: Rasmus Söderström Nylander and Erik Lidbjörk.
# Date: 2024.

import os
from ngram import NGramModel
from data import DataSource, DataSourceNTComments
from common import context_and_keystrokes
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train word prediction model.', usage='\n* -m Model file path. -d Dataset directory path. -n Number of datapoints to read.')
    parser.add_argument('-m', type=str, default='./model.txt', help='Model file path')
    parser.add_argument('-d', type=str, default='./data', help='Dataset directory path')
    parser.add_argument('-n', type=int, default=-1, help='Number of datapoints to read. Set to -1 to read whole dataset.')
    arguments = parser.parse_args()
    model_path = arguments.m
    data_path = arguments.d
    num_datapoints = arguments.n
    
    if os.path.isfile(model_path):
        print('Loading...')
        model = NGramModel.load(model_path)
        print('Done loading.')
    else:
        if data_path in ["data_nytimes", "./data_nytimes", "./data_nytimes/", ".\\data_nytimes", ".\\data_nytimes\\"]:
            source = DataSourceNTComments(data_path)
            sentences = lambda: source.sentences()
        else:
            source = DataSource(data_path)
            sentences = lambda: source.sentences(num_datapoints)
        print('Teaching...')
        model = NGramModel(3)
        for sentence in sentences():
            model.learn(sentence)
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
