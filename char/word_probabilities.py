# Author: Erik Lidbjörk.
# Date: 2024.

import numpy as np

"""
Interface for getting most likely words given some context of words.
"""
class WordProbabilities:
    """
    input_str: Unsanitized input string typed by the user. 
    n: Number of words and probabilities (int) to be returned by the method.

    Returns: 
    A tuple of two lists. One containing words (strings) ranked in order of highest 
    to lowest probabilities. Second list is corresponding probabilities. 
    """
    def most_likely_words(self, input_str: str, n: int) -> tuple[list[str], list[float]]:
        pass


"""
Demonstrate class behaviour with mock data.
"""
class ExampleWords(WordProbabilities):
    word_lists = [['Harry', 'Dumbledore', 'Spell', 'Wand'],
                  ['Ron', 'Snape', 'Door', 'and'],
                  ['to', 'for', 'Gryffindor', 'Slytherin']]
    prob_lists = [[.5, .3, .05, .01],
                  [.7, .1, .01, .01],
                  [.25, .2, .01, .0001]]

    def most_likely_words(self, input_str: str, n: int) -> tuple[list[str], list[float]]:
        rng = np.random.default_rng() 
        if len(input_str) == 0:
            return [], []
        return self.word_lists[rng.integers(len(self.word_lists))][:n], self.prob_lists[rng.integers(len(self.prob_lists))][:n]