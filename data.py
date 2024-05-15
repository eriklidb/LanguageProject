# Authors: Rasmus Söderström Nylander and Erik Lidbjörk.
# Date: 2024.

import os
import re
import csv
from tqdm import tqdm
from chardet import detect


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
        encoding_type = None
        for path in self._paths:
            if not encoding_type:
                encoding_type = self.get_encoding_type(path)
            with open(path, 'r', encoding=encoding_type) as f:
                for line in f.readlines():
                    line = line.strip()
                    sentences = split(line)
                    for sentence in sentences:
                        sentence = clean(sentence)
                        yield sentence
        return
    
    @staticmethod
    def get_encoding_type(path: str):
        encoding_type = None
        with open(path, 'rb') as f:
            encoding_type = detect(f.read())['encoding']
        return encoding_type
        
class DataSourceNTComments(DataSource):
    def sentences(self, max_comments):
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
            if not s or s.startswith(' ') or '<' in s or '>' in s or 'href=' in s or 'text=' in s or 'target=' in s or '&amp' in s:
                continue
            yield s
        return

    @staticmethod
    def clean(phrase):
        phrase = re.sub('<br/>', ' ', phrase) # Replace linebreak HTML symbol.
        return clean(phrase)
        

def split(line):
    split = re.split('\\.(?=(\\s+[A-Z])|$)', line)
    for s in split:
        # Because re.split behaves weirdly and includes lookahead and None
        if not s or s.startswith(' '):
            continue
        yield s
    return


def clean(phrase):
    phrase = phrase.strip().lower()
    phrase = re.sub('[^a-z\'\\s]', '', phrase)
    phrase = re.sub('\\s+', ' ', phrase)
    return phrase

