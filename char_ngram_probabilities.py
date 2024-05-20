# Author: Erik Lidbjörk and Rasmus Nylander.
# Date: 2024.

from word_probabilities import WordProbabilities
from ngram import NGramModel
from chardata import context_and_keystrokes

class CharNGramProbabilities(WordProbabilities):
    def __init__(self, model_path):
        self._model = NGramModel.load(model_path)

    def most_likely_words(self, input_str: str, n: int) -> tuple[list[str], list[float]]:
        #if len(input_str) == 0:
        #    return [], []
        context, keystrokes = context_and_keystrokes(input_str)
        return self._model.completions(context, keystrokes, n)

