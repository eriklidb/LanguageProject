# Authored by Patrik Johansson, 2024

import os
import argparse
from data import DataSource, DataSourceNTComments
import numpy as np
import math


def generate_sets(path, output_dir=None):
    if output_dir is None:
        output_dir = ''
    #train, val, test are supposed to add up to 1.
    #not the most efficient solution and haven't tested it but should work.
    samples = []
    with open(path, 'r', encoding ='utf-8') as f:
        for line in f:
            samples.append(line)
    rand = np.random.permutation([x for x in range(len(samples))])
    max = int(len(rand))

    traininterval = rand[:math.floor(max*0.7)]
    valinterval = rand[math.floor(max*0.7):math.floor(max*(0.7+0.2))]
    testinterval = rand[math.floor(max*(0.7+0.2)):]
    trainingset = [samples[x] for x in traininterval]
    validationset = [samples[x] for x in valinterval]
    testingset = [samples[x] for x in testinterval]
    

    train_path = os.path.join(output_dir, 'train.txt')
    val_path = os.path.join(output_dir, 'val.txt')
    test_path = os.path.join(output_dir, 'test.txt')

    with open(train_path, 'w', encoding='utf-8') as f:
        for i in trainingset:
            f.writelines(i)

    with open(val_path, 'w', encoding='utf-8') as f:
        for i in validationset:
            f.writelines(i)

    with open(test_path, 'w', encoding='utf-8') as f:
        for i in testingset:
            f.writelines(i)


def main():
    parser = argparse.ArgumentParser(description='Split samples file into testing, training and validation sets', usage='\n* -m Model file path. -d Dataset directory path. -n Number of datapoints to read.')
    parser.add_argument('-d', type=str, default='./data', help='Dataset directory path')
    parser.add_argument('-o', type=str, help='Output directory')
    parser.add_argument('-n', type=int, help='Number of datapoints to read. Default behavior is reading all the data.')
    arguments = parser.parse_args()

    out_dir = arguments.o
    data_path = arguments.d

    if 'nyt' in data_path.lower():
        source = DataSourceNTComments(data_path, arguments.n)
    else:
        source = DataSource(data_path, arguments.n)

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    samples_path = os.path.join(out_dir, 'samples.txt')
    source.save_samples(samples_path)
    generate_sets(samples_path, out_dir)




if __name__ == '__main__':
    main()
