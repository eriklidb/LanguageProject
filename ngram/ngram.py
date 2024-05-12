# Authored 2024.5 by Rasmus Nylander (nylanderDev)

import re
import numpy as np


class NGramModel:
    def __init__(self, n=2):
        self._n = n
        self._trie = FreqTrie()
        self._w2i = {}
        self._i2w = []
        self._ngram_stores = {}
        self._UNKNOWN = '<UNKNOWN>'
        self._START = '<START>'
        self._SPECIAL_WORDS = [self._UNKNOWN, self._START]
        self._w2i[self._UNKNOWN] = len(self._i2w)
        self._i2w.append(self._UNKNOWN)
        self._w2i[self._START] = len(self._i2w)
        self._i2w.append(self._START)
        for i in range(1, n+1):
            self._ngram_stores[i] = NGramStore(i)


    def freq(self, ngram):
        n = len(ngram)
        return self._ngram_stores[n].freq(ngram)

    
    def learn(self, sentence):
        words = sentence.split()
        padding = [self._START] * (self._n - 1)
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


    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            vocab_size = len(self._i2w)
            header = f'{self._n} {vocab_size}\n'
            f.writelines(header)
            for i in range(vocab_size):
                idx = i
                word = self._i2w[idx]
                freq = self._ngram_stores[1].freq([idx])
                f.writelines(f'{idx} {word} {freq}\n')
            for k in range(2, self._n+1):
                kgrams = self._ngram_stores[k].all_ngrams()
                count = len(kgrams)
                f.writelines(f'{count}\n')
                for kgram, freq in kgrams:
                    kgram_str = ' '.join(map(str, kgram))
                    f.writelines(f'{kgram_str} {freq}\n')


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


    def completions(self, context, keystrokes, n=1):
        context = self.transform_input(context) 
        if self._n == 1:
            context = []
        elif len(context) > self._n - 1:
            context = context[-(self._n - 1):]

        print('context', context)
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

        debug = list(zip(weights, candidates))
        debug.sort(reverse=True)
        ddd = min(len(debug), 3)
        print(debug[:ddd])
        if tot > 0:
            prob = np.array(weights)
            completions = np.random.choice(candidates,\
                    size=n,\
                    p=prob,\
                    replace=False)
            return completions 
        return []

        
    def transform_input(self, phrase):
        phrase = self.clean(phrase)
        words = phrase.split()
        print('words', words)
        words = map(lambda w: w if w in self._w2i else self._UNKNOWN, words)
        padding = [self._START] * (self._n - 1)
        words = padding + list(words)
        indices = list(map(lambda w: self._w2i[w], words))
        return indices

    def clean(self, phrase):
        phrase = phrase.strip().lower()
        phrase = re.sub('[^a-z\'\s]', '', phrase)
        phrase = re.sub('\s+', ' ', phrase)
        return phrase


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


class FreqTrie:
    def __init__(self):
        self._root = FreqTrieNode()
    

    def add_word(self, word, n=1):
        self._root.add_suffix(word, n)


    def get_words(self, prefix):
        subroot = self._root.child(prefix, create=False)
        if subroot is None:
            return (0, [], [])

        nodes = subroot.descendants()
        total_freqs = subroot.subfreq()
        words = []
        freqs = []

        for node in nodes:
            if node.freq() > 0:
                words.append(node.word())
                freqs.append(node.freq())

        return (total_freqs, words, freqs)



class FreqTrieNode:
    def __init__(self, word=None):
        if word is None:
            self._word = ''
        else:
            self._word = word
        self._freq = 0
        self._subfreq = 0
        self._children = {}

    
    def add_suffix(self, suffix, n=1):
        if suffix == '':
            self.increment_freq(n)
        else:
            self.increment_subfreq(n)
            head = suffix[0]
            tail = suffix[1:]
            self.child(head).add_suffix(tail)
    

    def increment_freq(self, n=1):
        self._freq += n

    
    def increment_subfreq(self, n=1):
        self._subfreq += n


    def word(self):
        return self._word


    def freq(self):
        return self._freq


    def subfreq(self):
        return self._subfreq


    def child(self, suffix, create=True):
        if suffix == '':
            return self
        else:
            head = suffix[0]
            tail = suffix[1:]
            if head not in self._children:
                if not create:
                    return None
                child_word = self._word + head
                child = FreqTrieNode(child_word)
                self._children[head] = child
            return self._children[head].child(tail)


    def descendants(self):
        nodes = []
        for child in self._children.values():
            nodes.append(child)
            nodes.extend(child.descendants())
        return nodes
