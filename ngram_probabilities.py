# Author: Erik Lidbjörk and Rasmus Nylander.
# Date: 2024.

from word_probabilities import WordProbabilities
from ngram import NGramModel
from common import context_and_keystrokes

class NGramProbabilities(WordProbabilities):
    def __init__(self, model_path):
        self._model = NGramModel.load(model_path)

    def most_likely_words(self, current_words: list[str], n: int) -> tuple[list[str], list[float]]:
        text = ' '.join(current_words)
        context, keystrokes = context_and_keystrokes(text)
        return self._model.completions(context, keystrokes, n)

