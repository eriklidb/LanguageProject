import os
import re
import random


class Special:
    PADDING = '<P>'
    UNKNOWN = '<U>'
    START = '<S>'
    
    def all():
        return [Special.PADDING, Special.UNKNOWN, Special.START]

    
    def size():
        return len(Special.all())


class DataSource:
    def __init__(self, path):
        if os.path.isdir(path):
            self._path = path
            vocab_name = 'vocab.txt'
            self._vocab_path = vocab_path = os.path.join(self._path, vocab_name)
            
            stale_vocab = False
            if os.path.isfile(self._vocab_path):
                vocab_time = os.path.getmtime(self._vocab_path)
            else:
                vocab_time = 0

            paths = []
            for root, _, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(self._path, filename) 
                    if filepath != vocab_path:
                         paths.append(filepath)
                    if os.path.getmtime(filepath) > vocab_time:
                        stale_vocab = True
            self._paths = paths
            if stale_vocab:
                self._ensure_vocab()
        else:
            raise ValueError(f'Path {path} does not exist')


    def _ensure_vocab(self):
        vocab = set()
        for sentence in self.sentences():
            sentence = clean(sentence)
            words = sentence.split()
            for word in words:
                vocab.add(word)
        with open(self._vocab_path, 'w', encoding='utf-8') as f:
            for word in vocab:
                f.writelines(f'{word}\n')

    def vocab(self):
        with open(self._vocab_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                word = line.strip()
                yield word



    def sentences(self):
        for path in self._paths:
            with open(path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    sentences = split(line)
                    for sentence in sentences:
                        sentence = clean(sentence)
                        yield sentence


    def labeled_samples(self, min_context_len, max_context_len):
        for sentence in self.sentences():
            words = sentence.split()
            padding = [Special.START] * max_context_len
            words = padding + words
            for i in range(max_context_len, len(words)):
                kgram_label = words[i]
                for k in range(min_context_len, max_context_len):
                    kgram_context = ' '.join(words[i-k:i])
                    yield kgram_context, kgram_label

        """
        random.seed(seed)
        for sentence in self.sentences():
            words = sentence.split()

            samples_per_sentence = int(sample_ratio * len(words))
            samples_per_sentence = max(1, samples_per_sentence)
            for i in range(samples_per_sentence):
                ctx_len = int(random.random() * len(words))
                label = words[ctx_len]
                ctx = " ".join(words[:ctx_len])

                splits_per_sample = int(split_ratio * len(label))
                splits_per_sample = max(1, splits_per_sample)
                for j in range(splits_per_sample):

                    split = int(random.random() * len(label))
                    partial = label[:split]
                    sample = f'{ctx} {partial}'
                    yield sample, label
        """


    def save_samples(self, path, min_ctx_len, max_ctx_len):
        with open(path, 'w', encoding='utf-8') as f:
            for sample, label in self.labeled_samples(min_ctx_len, max_ctx_len):
                f.writelines(f'{sample},{label}\n')


def split(line):
    split = re.split('\.(?=(\s+[A-Z])|$)', line)
    for s in split:
        # Because re.split behaves weirdly and includes lookahead and None
        if not s or s.startswith(' '):
            continue
        yield s
    return


def clean(phrase):
    phrase = phrase.strip().lower()
    phrase = re.sub('[^a-z\'\s]', '', phrase)
    phrase = re.sub('\s+', ' ', phrase)
    return phrase


def context_and_keystrokes(text):
    if text.endswith(' '):
        keystrokes = ''
        context = text.strip()
    else:
        split = text.split()
        if len(split) > 0:
            keystrokes = text.split()[-1]
        else:
            keystrokes = ''
        context = text[:-len(keystrokes)]
    return context, keystrokes


