# Authors: Rasmus Söderström Nylander and Erik Lidbjörk.
# Date: 2024.

import os
import re
import csv
from tqdm import tqdm
from chardet import detect


class Special:
    PADDING = '<P>'
    UNKNOWN = '<U>'
    START = '<S>'
    
    def all():
        return [Special.PADDING, Special.UNKNOWN, Special.START]

    
    def size():
        return len(Special.all())


class DataSource:
    def __init__(self, path, num_datapoints):
        self.num_datapoints = num_datapoints
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
            sentence = self.clean(sentence)
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
        encoding_type = None
        for path in self._paths:
            if not encoding_type:
                encoding_type = self.get_encoding_type(path)
            with open(path, 'r', encoding=encoding_type) as f:
                for line in f.readlines():
                    line = line.strip()
                    sentences = self.split(line)
                    for sentence in sentences:
                        sentence = self.clean(sentence)
                        yield sentence


    def labeled_samples(self):
        for sentence in self.sentences():
            sentence = f'{Special.START} {sentence}'
            words = sentence.split()
            sentence_len = len(words)
            for i in range(1, sentence_len):
                kgram_label = words[i]
                kgram_context = ' '.join(words[:i])
                yield kgram_context, kgram_label


    def labeled_samples_batch(self, batch_size, discard_trailing=False):
        batch_size = max(1, batch_size)
        batch_ctx = []
        batch_labels = []
        for ctx, label in self.labeled_samples():
            batch_ctx.append(ctx)
            batch_labels.append(label)
            if len(batch_labels) == batch_size:
                yield batch_ctx, batch_labels
                batch_ctx = []
                batch_labels = []
        if len(batch_labels) > 0 and not discard_trailing:
            yield batch_ctx, batch_labels


    def save_samples(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            for sample, label in self.labeled_samples():
                f.writelines(f'{sample},{label}\n')

    
    @staticmethod
    def get_encoding_type(path: str):
        encoding_type = None
        with open(path, 'rb') as f:
            encoding_type = detect(f.read())['encoding']
        return encoding_type

    @staticmethod
    def split(line):
        split = re.split('\\.(?=(\\s+[A-Z])|$)', line)
        for s in split:
            # Because re.split behaves weirdly and includes lookahead and None
            if not s or s.startswith(' '):
                continue
            yield s
        return


    @staticmethod
    def clean(phrase):
        phrase = phrase.strip().lower()
        phrase = re.sub('[^a-z\'\\s]', '', phrase)
        phrase = re.sub('\\s+', ' ', phrase)
        return phrase


class DataSourceNTComments(DataSource):
    def sentences(self):
        max_comments = self.num_datapoints
        #processed_comments = 0
        print("Reading New York Times dataset.")
        processed_comments = 0
        encoding_type = 'utf8'
        for path in self._paths:
            # Look only at comment files.
            if "Comments" not in path:
                continue
            if not encoding_type:
                encoding_type = self.get_encoding_type(path)
            print("Reading", path)
            with open(file=path, mode='r', encoding=encoding_type) as f:
                reader = csv.reader(f)
                fields = next(reader) # Column names.
                comment_body_index = fields.index('commentBody')
                for row in tqdm(reader):
                    if max_comments and processed_comments >= max_comments:
                        return
                    comment = row[comment_body_index].strip()
                    sentences = self.split(comment)
                    for sentence in sentences:
                        sentence = self.clean(sentence)
                        yield sentence
                    processed_comments += 1
            print("Finished reading", path)
        return


    @staticmethod
    def split(line):
        split = re.split('\\.(?=(\\s+[A-Z])|$)', line)
        for s in split:
            # Remove HTML tags and other junk.
            if not s or s.startswith(' ') or 'href=' in s or 'text=' in s or 'target=' in s or '&amp' in s or 'http' in s or "www." in s:
                continue
            yield s
        return

    @staticmethod
    def clean(phrase):
        phrase = re.sub('<br/>', ' ', phrase) # Replace linebreak HTML symbol.
        return DataSource.clean(phrase)
        

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


       