import os
import re


class DataSource:
    def __init__(self, path):
        if os.path.isdir(path):
            paths = []
            for root, _, filenames in os.walk(path):
                for filename in filenames:
                    paths.append(os.path.join(root, filename))
        elif os.path.isfile(path):
            paths = [path]
        else:
            raise ValueError(f'Path {path} does not exist')
        self._paths = paths


    def sentences(self):
        for path in self._paths:
            with open(path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    sentences = split(line)
                    for sentence in sentences:
                        sentence = clean(sentence)
                        yield sentence
        return


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

