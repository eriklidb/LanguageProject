# Author: Erik Lidbjörk and Rasmus Nylander.
# Date: 2024.

from word_probabilities import WordProbabilities
from neural import NeuralPredictor
from data import context_and_keystrokes

class NeuralProbabilities(WordProbabilities):
    def __init__(self, model_path):
        self._model = NeuralPredictor.load(model_path)

    def most_likely_words(self, input_str: str, n: int) -> tuple[list[str], list[float]]:
        context, keystrokes = context_and_keystrokes(input_str)
        context = f'<S> {context}'
        return self._model.completions(context, keystrokes, n)

