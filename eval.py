import os
from ngram import NGramModel
from neural import NeuralPredictor
import argparse
from data import Special, DataSource, DataSourceNTComments, DataLoader, context_and_keystrokes


def predict(model, context, k):
    prediction, _ = model.completions(context, '', n=k, deterministic=True)
    if len(prediction) <= 0:
        return []
    else:
        return prediction


def evaluate(model, data_src, k=None):
    if k is None or k == []:
        k = [1]
    max_k = max(k)
    print(k)
    keystrokes_total = 0
    keystrokes_saved = [0] * len(k)
    total = 0
    correct = [0] * len(k)
    acc = [0] * len(k)
    for ctx, label in data_src.labeled_samples():
        pred = predict(model, ctx, max_k)
        ctx_len = len(ctx.split()) - 1
        keystrokes_total += len(label)
        for i in range(len(k)):
            if label in pred[:k[i]]:
                #print('--- correctly predicted:', label)
                correct[i] += 1
                keystrokes_saved[i] += (len(label) - ctx_len)

        total += 1
        if total % 100 == 0:
            print(f'context: {ctx}')
            print(f'actual: {label}')
            print(f'predicted: {pred}')
            for i in range(len(acc)):
                print(f'SO FAR: accuracy (top {k[i]}):\t {correct[i] / total}\t [{correct[i]} out of {total}]')
            print('---')
            for i in range(len(acc)):
                print(f'Assuming char-level: % keystrokes saved (top {k[i]}):\t {keystrokes_saved[i] / keystrokes_total}\t [{keystrokes_saved[i]} out of {keystrokes_total}]')
            
            print(f'evaled {total} datapoints')
    for i in range(len(k)):
        acc[i] = correct[i] / total
    return acc, correct, total



def main():
    parser = argparse.ArgumentParser(description='Evaluate word prediction model.', usage='\n* -m Model path. -d Validation dataset path.')
    parser.add_argument('-m', type=str, default='./model.txt', help='Model file path')
    parser.add_argument('-d', type=str, default='./data', help='Dataset directory path')
    arguments = parser.parse_args()
    model_path = arguments.m
    data_path = arguments.d
    print('Running!')
    if os.path.isfile(model_path):
        model = NGramModel.load(model_path)
    elif os.path.isdir(model_path):
        model = NeuralPredictor.load(model_path)
    else:
        print('Could not find model')
        exit(-1)
    
    data = DataLoader(data_path)
    k = [1, 3, 5]
    print(k)
    acc, correct, total = evaluate(model, data, k=k)
    for i in range(len(acc)):
        print(f'accuracy (top {k[i]}):\t {acc[i]}\t [{correct[i]} out of {total}]')


if __name__ == '__main__':
    main()


