# Authored by Patrik Johansson, 2024

import os
import argparse
from data import DataSource, DataSourceNTComments
import numpy as np
import math
import random

class Samples:
    def __init__(self):
        self._strata = {}
        self._labels = []
        self._samples = []
        self._samples_left = []
        self._stratified = True


    def load(self, path):
        with open(path, 'r', encoding ='utf-8') as f:
            for line in f:
                label = line.split(',')[1].rstrip()
                idx = len(self._samples)
                if label not in self._strata:
                    self._labels.append(label)
                    self._strata[label] = set()
                self._strata[label].add(idx)
                self._samples_left.append(idx)
                self._samples.append(line)


    def _remove_strata(self, label):
        del self._strata[label]
        self._labels.remove(label)

    
    def _pop_sample(self, i):
        idx = self._samples_left[i]
        if not self._stratified:
            temp = self._samples_left[-1]
            self._samples_left[i] = temp
            self._samples_left.pop()
        else:
            self._samples_left[i] = None
        
        sample = self._samples[idx]
        label = sample.split(',')[1].rstrip()
        self._strata[label].discard(idx)
        if len(self._strata[label]) == 0:
            self._remove_strata(label)
        self._samples[idx] = None
        return sample


    def random_samples(self, n):
        if self._stratified:
            self._stratified = False
            self._samples_left = list(filter(lambda x: x, self._samples_left))
        samples = []
        for i in range(n):
            ii = int(random.random() * len(self._samples_left))
            sample = self._pop_sample(ii)
            samples.append(sample)
        return samples


    def stratified_samples(self, n):
        if not self._stratified:
            raise Exception('Cannot sample stratified after sampling random')
        samples = []
        for i in range(n):
            ii = int(random.random() * len(self._labels))
            label = self._labels[ii]
            strata = self._strata[label]
            iii = strata.pop()
            sample = self._pop_sample(iii)
            samples.append(sample)
        return samples



def generate_sets(path, n, output_dir=None):
    if output_dir is None:
        output_dir = ''
    #train, val, test are supposed to add up to 1.
    #not the most efficient solution and haven't tested it but should work.
    # train: 0.7
    # val:   0.2
    # test:  0.1
    samples = Samples()
    samples.load(path)
    n_train = int(0.7 * n)
    n_val = int(0.2 * n)
    n_test = n - n_train - n_val

    train_path = os.path.join(output_dir, 'train.txt')
    val_path = os.path.join(output_dir, 'val.txt')
    test_path = os.path.join(output_dir, 'test.txt')

    with open(train_path, 'w', encoding='utf-8') as f:
        for i in samples.stratified_samples(n_train):
            f.writelines([i])

    with open(val_path, 'w', encoding='utf-8') as f:
        for i in samples.stratified_samples(n_val):
            f.writelines([i])

    with open(test_path, 'w', encoding='utf-8') as f:
        for i in samples.stratified_samples(n_test):
            f.writelines([i])


def main():
    parser = argparse.ArgumentParser(description='Split samples file into testing, training and validation sets', usage='\n* -m Model file path. -d Dataset directory path. OR -s Sample file path. -n Number of datapoints to read.')
    parser.add_argument('-d', type=str, help='Dataset directory path')
    parser.add_argument('-s', type=str, help='Sample file path')
    parser.add_argument('-o', type=str, help='Output directory')
    parser.add_argument('-n', type=int, help='Number of datapoints to read. Default behavior is reading all the data.')
    arguments = parser.parse_args()

    out_dir = arguments.o
    data_path = arguments.d
    samples_path = arguments.s
    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    if not samples_path:
        if 'nyt' in data_path.lower():
            source = DataSourceNTComments(data_path, 10 * arguments.n)
        else:
            source = DataSource(data_path, arguments.n)
        samples_path = os.path.join(out_dir, 'samples.txt')
        source.save_samples(samples_path)

    
    generate_sets(samples_path, arguments.n, out_dir)




if __name__ == '__main__':
    main()
