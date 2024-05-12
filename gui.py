# Author: Erik Lidbjörk.
# Date: 2024.

import tkinter as tk
from word_probabilities import WordProbabilities, ExampleWords
from ngram_probabilities import NGramProbabilities

class Window(tk.Tk):
    word_probabilities: WordProbabilities
    num_words_displayed: int
    background: str = 'white'

    label_header_word: tk.Label
    label_header_prob: tk.Label
    labels_word: list[tk.Label]
    labels_prob: list[tk.Label]
    text_input: tk.Text

    def __init__(self, word_probabilities: WordProbabilities = ExampleWords(), num_words_displayed: int = 10) -> None:
        super().__init__()
        self.word_probabilities = word_probabilities
        self.num_words_displayed = num_words_displayed

        self.title("Word Predictor")
        #self.geometry('400x200') 
        self.configure(background='white')

        self.label_header_word = tk.Label(self, text = "Words", background=self.background, font=('Helvetica', 16, 'bold')) 
        self.label_header_word.grid(sticky="W", row=0, column=0)
        self.label_header_prob = tk.Label(self, text = "Probabilities", background=self.background, font=('Helvetica', 16, 'bold')) 
        self.label_header_prob.grid(sticky="W", row=0, column=1)

        self.labels_word = []
        self.labels_prob = []
        for i in range(self.num_words_displayed):
            label_word = tk.Label(self, text = "", background=self.background) 
            label_word.grid(sticky="W", row=i+1, column=0)
            label_word.bind("<Button-1>", self.handle_word_press)
            self.labels_word.append(label_word)

            label_prob = tk.Label(self, text = "", background=self.background) 
            label_prob.grid(sticky="W", row=i+1, column=1)
            self.labels_prob.append(label_prob)

        self.text_input = tk.Text(self, height = 10, width = 60) 
        self.text_input.grid(sticky="W", row=self.num_words_displayed + 2, column=0, columnspan=2)
        self.text_input.bind("<KeyRelease>", self.handle_keystroke)

    def handle_keystroke(self, _event: tk.Event) -> None:
        self.update_displayed_words()

    def handle_word_press(self, event: tk.Event) -> None:
        pressed_word = event.widget.cget("text")
        if pressed_word != "":
            self.correct_text(pressed_word)

    def update_displayed_words(self) -> None:
        input_str = self.text_input.get(1.0, "end-1c")
        words, probs = self.word_probabilities.most_likely_words(input_str, self.num_words_displayed)
        for i in range(self.num_words_displayed):
            text_word = ""
            text_prob = ""
            if i < len(words):
                text_word = words[i]
                text_prob = str(probs[i])
            self.labels_word[i].config(text = text_word)
            self.labels_prob[i].config(text = text_prob)

    def correct_text(self, pressed_word: str) -> None:
        pressed_word = pressed_word + " "
        input_str = self.text_input.get(1.0, "end-1c")
        if len(input_str) == 0:
            self.text_input.insert(1.0, pressed_word)
        elif input_str[-1].isspace():
            self.text_input.insert("end-1c", pressed_word)
        else:
            last_space_index = input_str.rfind(" ")
            last_newline_index = input_str.rfind("\n")
            replace_index = max(last_space_index, last_newline_index)
            new_input_str = input_str[:replace_index + 1] + pressed_word
            self.text_input.delete(1.0, "end-1c")
            self.text_input.insert(1.0, new_input_str)
        self.update_displayed_words()
            
# Start the event loop.
if __name__ == '__main__':
    probabilities = NGramProbabilities('model.txt')
    window = Window(word_probabilities=probabilities)
    window.mainloop()
