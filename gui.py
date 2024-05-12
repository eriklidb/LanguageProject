# Author: Erik Lidbjörk.
# Date: 2024.

import tkinter as tk
from word_probabilities import WordProbabilities, ExampleWords

class Window(tk.Tk):
    word_probabilities: WordProbabilities
    num_words_displayed: int
    buffer_words: list[str] = []
    background: str = 'white'

    def __init__(self, word_probabilities=ExampleWords(), num_words_displayed=10):
        super().__init__()
        self.word_probabilities = word_probabilities
        self.num_words_displayed = num_words_displayed

        self.title("Word Predictor")
        #self.geometry('400x200') 
        self.configure(background='white')

        self.header_label_word = tk.Label(self, text = "Words", background=self.background, font=('Helvetica', 16, 'bold')) 
        self.header_label_word.grid(sticky="W", row=0, column=0)
        self.header_label_prob = tk.Label(self, text = "Probabilities", background=self.background, font=('Helvetica', 16, 'bold')) 
        self.header_label_prob.grid(sticky="W", row=0, column=1)

        self.word_labels = []
        self.prob_labels = []
        for i in range(self.num_words_displayed):
            word_label = tk.Label(self, text = "", background=self.background) 
            word_label.grid(sticky="W", row=i+1, column=0)
            self.word_labels.append(word_label)

            prob_label = tk.Label(self, text = "", background=self.background) 
            prob_label.grid(sticky="W", row=i+1, column=1)
            self.prob_labels.append(prob_label)

        self.input_txt = tk.Text(self, height = 10, width = 60) 
        self.input_txt.grid(sticky="W", row=self.num_words_displayed + 2, column=0, columnspan=2)
        self.input_txt.bind("<KeyRelease>", self.handle_keystroke)

    def handle_keystroke(self, event):
        inp = self.input_txt.get(1.0, "end-1c")
        self.buffer = inp.split()
        self.update_displayed_words()

    def update_displayed_words(self):
        words, probs = self.word_probabilities.most_likely_words(self.buffer, self.num_words_displayed)
        for i in range(self.num_words_displayed):
            text_word = ""
            text_prob = ""
            if i < len(words):
                text_word = words[i]
                text_prob = str(probs[i])
            self.word_labels[i].config(text = text_word)
            self.prob_labels[i].config(text = text_prob)

# Start the event loop.
if __name__ == '__main__':
    window = Window()
    window.mainloop()
