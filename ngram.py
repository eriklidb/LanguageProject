# Authored 2024.5 by Rasmus Nylander (nylanderDev)

import re
import numpy as np
from data import Special, clean
from trie import FreqTrie

class NGramModel:
    def __init__(self, n=2):
        self._n = n
        self._trie = FreqTrie()
        self._w2i = {}
        self._i2w = []
        self._ngram_stores = {}
        self._SPECIAL_WORDS = [Special.UNKNOWN, Special.START]
        self.add_word(Special.UNKNOWN)
        self.add_word(Special.START)
        for i in range(1, n+1):
            self._ngram_stores[i] = NGramStore(i)


    def freq(self, ngram):
        n = len(ngram)
        return self._ngram_stores[n].freq(ngram)

    
    def learn(self, sentence):
        words = sentence.split()
        padding = [Special.START] * (self._n - 1)
        words = padding + words
        for i in range(len(words)):
            word = words[i]
            self.add_word(word)
            for k in range(self._n):
                if i < k:
                    continue
                kgram_words = words[i-k:i+1]
                kgram = list(map(lambda w: self._w2i[w], kgram_words))
                self.add_ngram(kgram)


    def learn_sample(self, context, label):
        words = f'{context} {label}'.split()
        for word in words:
            self.add_word(word)
        kgram = list(map(lambda w: self._w2i[w], words))
        self.add_ngram(kgram)


    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            vocab_size = len(self._i2w)
            header = f'{self._n} {vocab_size}\n'
            f.writelines([header])
            for i in range(vocab_size):
                idx = i
                word = self._i2w[idx]
                freq = self._ngram_stores[1].freq([idx])
                f.writelines([f'{idx} {word} {freq}\n'])
            for k in range(2, self._n+1):
                kgrams = self._ngram_stores[k].all_ngrams()
                count = len(kgrams)
                f.writelines(f'{count}\n')
                for kgram, freq in kgrams:
                    kgram_str = ' '.join(map(str, kgram))
                    f.writelines([f'{kgram_str} {freq}\n'])


    def load(path):
        with open(path, 'r', encoding='utf-8') as f:
            headers = f.readline().strip().split(' ')
            _n = int(headers[0])
            model = NGramModel(_n)
            vocab_size = int(headers[1])
            model._i2w = [None] * vocab_size
            for i in range(vocab_size):
                parts = f.readline().strip().split(' ')
                idx = int(parts[0])
                word = parts[1]
                freq = int(parts[2])
                model._w2i[word] = idx
                model._i2w[idx] = word
                model.add_ngram([idx], freq)
                model.add_word(word, freq)
            for k in range(2, _n+1):
                headers = f.readline().strip().split(' ')
                kgram_count = int(headers[0])
                for i in range(kgram_count):
                    parts = f.readline().strip().split(' ')
                    kgram = list(map(int, parts[:k]))
                    freq = int(parts[k])
                    model.add_ngram(kgram, freq)
            return model


    def add_ngram(self, ngram, freq=1):
        n = len(ngram)
        if n > self._n or n <= 0:
            raise ValueError(f'Expected 0<N<={self._n} for NGram, got {n}-gram')
        
        self._ngram_stores[n].add_ngram(ngram, freq)


    def add_word(self, word, n=1):
        if word not in self._w2i:
            idx = len(self._i2w)
            self._w2i[word] = idx
            self._i2w.append(word)
        if word not in self._SPECIAL_WORDS:
            self._trie.add_word(word, n)


    def completions(self, context, keystrokes, n=1, deterministic=True):
        context = self.transform_input(context) 
        if self._n == 1:
            context = []
        elif len(context) > self._n - 1:
            context = context[-(self._n - 1):]

        tot, candidates, _ = self._trie.get_words(keystrokes)
        if n < 0:
            n = len(candidates) 
        else:
            n = min(n, len(candidates))
        
        freqs = {}
        probs = {}
        for k in range(len(context)+1):
            freqs[k] = [0] * len(candidates)
            probs[k] = [0] * len(candidates)
            tot_freq = 0
            if k == 0:
                subctx = []
            else:
                subctx = context[-k:]
            for i in range(len(candidates)):
                cand = candidates[i]
                idx = self._w2i[cand]
                ngram = subctx + [idx]
                freq = self.freq(ngram)
                freqs[k][i] = freq
                tot_freq += freq
            for i in range(len(candidates)):
                if tot_freq != 0:
                    probs[k][i] = freqs[k][i] / tot_freq
                else:
                    probs[k][i] = 1 / len(candidates)

        # These should be lambda parameters FIXME
        BASE_WEIGHT = 0.00001
        weights = [BASE_WEIGHT] * len(candidates)
        N_WEIGHT = 0.99
        DEFAULT_WEIGHT = 0.01 - BASE_WEIGHT
        total_weight = 0
        for i in range(len(candidates)):
            for k in range(len(context)+1):
                if k == self._n - 1:
                    weights[i] += N_WEIGHT * probs[k][i] 
                else:
                    weights[i] += DEFAULT_WEIGHT * probs[k][i]
                
            total_weight += weights[i]
        
        for i in range(len(candidates)):
            weights[i] /= total_weight
        
        if tot <= 0:
            return [], []
        elif deterministic:
            det = list(zip(weights, candidates))
            det.sort(reverse=True)
            probs, completions = list(zip(*det[:n]))
            completions = list(completions)
            probs = list(probs)
            return completions, probs
        else:
            prob = np.array(weights)
            indices = np.random.choice(len(candidates),\
                    size=n,\
                    p=prob,\
                    replace=False)
            sampled = list(map(lambda i: (weights[i], candidates[i]), indices))
            sampled.sort(reverse=True)
            probs, completions = list(zip(*sampled[:n]))
            completions = list(completions)
            probs = list(probs)
            return completions, probs

        
    def transform_input(self, phrase):
        phrase = clean(phrase)
        words = phrase.split()
        words = map(lambda w: w if w in self._w2i else Special.UNKNOWN, words)
        padding = [Special.START] * (self._n - 1)
        words = padding + list(words)
        indices = list(map(lambda w: self._w2i[w], words))
        return indices


class NGramStore:
    def __init__(self, n):
        if n <= 0:
            raise ValueError(f'Expected N>=1 for NGramStore, got {n}')
        self._n = n
        self._root = {}
        self._total = 0


    def add_ngram(self, ngram, freq=1):
        if len(ngram) != self._n:
            m = len(ngram)
            raise ValueError(f'Expected {self._n}-gram, got {m}-gram')
        stubgram = ngram[:-1]
        uni = ngram[-1]
        unigrams = self._stub_dict(stubgram)
        if uni not in unigrams:
            unigrams[uni] = 0
        unigrams[uni] += freq


    def _stub_dict(self, stubgram, create=True):
        curr = self._root
        for i in range(self._n - 1):
            idx = stubgram[i]
            if idx not in curr:
                if create:
                    curr[idx] = {}
                else:
                    return None
            curr = curr[idx]
        return curr


    def unigrams(self, stubgram):
        if len(stubgram) != self._n - 1:
            m = len(stubgram)
            raise ValueError(f'Expected {self._n - 1}-gram as stub, got {m}-gram')

        unigrams = self._stub_dict(stubgram)
        if unigrams is None:
            return []
        return unigrams.items()


    def freq(self, ngram):
        if len(ngram) != self._n:
            m = len(ngram)
            raise ValueError(f'Expected {self._n}-gram, got {m}-gram')
        unigrams = self._stub_dict(ngram[:-1])
        uni = ngram[-1]
        if uni in unigrams:
            return unigrams[ngram[-1]]
        else:
            return 0


    def all_ngrams(self):
        ngrams = []
        queue = [([], self._root)]
        while len(queue) > 0:
            stub, curr = queue.pop(0)
            if len(stub) == self._n:
                ngrams.append((stub, curr))
            else:
                for idx in curr:
                    next_ = curr[idx]
                    next_stub = stub.copy()
                    next_stub.append(idx)
                    queue.append((next_stub, next_))
        return ngrams

