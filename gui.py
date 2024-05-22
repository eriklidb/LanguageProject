# Author: Erik Lidbjörk.
# Date: 2024.

import tkinter as tk
import argparse
from word_probabilities import WordProbabilities 
from ngram_probabilities import NGramProbabilities
from char_ngram_probabilities import CharNGramProbabilities
from neural_probabilities import NeuralProbabilities
from char_neural_probabilities import CharNeuralProbabilities

class Window(tk.Tk):
    word_probabilities: dict[str, WordProbabilities] = {}
    current_model: str = ''
    num_words_displayed: int
    background: str = 'white'
    background_hover: str = 'yellow'
    background_active: str = 'green'

    label_header_word: tk.Label
    label_header_prob: tk.Label
    label_header_model: tk.Label
    labels_word: list[tk.Label]
    labels_prob: list[tk.Label]
    labels_model: dict[str, tk.Label]
    text_input: tk.Text

    def __init__(self, num_words_displayed: int = 20) -> None:
        super().__init__()
        self.num_words_displayed = num_words_displayed

        self.title("Word Predictor")
        #self.geometry('400x200') 
        self.configure(background='white')

        self.label_header_word = tk.Label(self, text = "Words", background=self.background, font=('Helvetica', 16, 'bold')) 
        self.label_header_word.grid(sticky="W", row=0, column=0)
        self.label_header_prob = tk.Label(self, text = "Probabilities", background=self.background, font=('Helvetica', 16, 'bold')) 
        self.label_header_prob.grid(sticky="W", row=0, column=1)
        self.label_header_prob = tk.Label(self, text = "Model", background=self.background, font=('Helvetica', 16, 'bold')) 
        self.label_header_prob.grid(sticky="W", row=0, column=2)

        self.labels_word = []
        self.labels_prob = []
        self.labels_model = {}
        for i in range(self.num_words_displayed):
            label_word = tk.Label(self, text="", background=self.background) 
            label_word.grid(sticky="W", row=i+1, column=0, ipadx=10)
            label_word.bind("<Button-1>", self.handle_word_press)
            label_word.bind("<Enter>", self.handle_enter_label)
            label_word.bind("<Leave>", self.handle_leave_label)
            self.labels_word.append(label_word)

            label_prob = tk.Label(self, text="", background=self.background) 
            label_prob.grid(sticky="W", row=i+1, column=1)
            self.labels_prob.append(label_prob)

        self.text_input = tk.Text(self, height = 15, width = 120) 
        self.text_input.grid(sticky="W", row=self.num_words_displayed + 2, column=0, columnspan=2)
        self.text_input.bind("<KeyRelease>", self.handle_keystroke)

    def insert_word_probability(self, key: str, wp: WordProbabilities) -> None:
        bg = self.background
        if not self.current_model:
            self.current_model = key
            bg = self.background_active
        label_model = tk.Label(self, text=key, background=bg, font=('Helvetica', 12)) 
        label_model.bind("<Button-1>", self.handle_model_press)
        label_model.bind("<Enter>", self.handle_enter_label)
        label_model.bind("<Leave>", self.handle_leave_label)
        self.labels_model[key] = label_model
        label_model.grid(sticky="W", row=len(self.labels_model), column=2)
        self.word_probabilities[key] = wp

    def handle_keystroke(self, _event: tk.Event) -> None:
        self.update_displayed_words()

    def handle_word_press(self, event: tk.Event) -> None:
        pressed_word = event.widget.cget("text")
        if pressed_word != "":
            self.correct_text(pressed_word)

    def handle_enter_label(self, event: tk.Event) -> None:
        if event.widget.cget("text") not in ["", self.current_model]:
            event.widget.config(background=self.background_hover)

    def handle_leave_label(self, event: tk.Event) -> None:
        if event.widget.cget("text") not in ["", self.current_model]:
            event.widget.config(background=self.background)

    def handle_model_press(self, event: tk.Event) -> None:
        self.update_model(event.widget.cget("text"))
        
    def update_displayed_words(self) -> None:
        input_str = self.text_input.get(1.0, "end-1c")
        words, probs = self.word_probabilities[self.current_model].most_likely_words(input_str, self.num_words_displayed)
        for i in range(self.num_words_displayed):
            text_word = ""
            text_prob = ""
            if i < len(words):
                text_word = words[i]
                text_prob = str(probs[i])
            self.labels_word[i].config(text = text_word)
            self.labels_prob[i].config(text = text_prob)

    def update_model(self, model: str) -> None:
        self.current_model = model
        for m, label in self.labels_model.items():
            if m == self.current_model:
                label.config(background=self.background_active)
            else:
                label.config(background=self.background)
        self.update_displayed_words()

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

    # Override.
    def mainloop(self) -> None:
        self.update_displayed_words()
        super().mainloop()
            
# Start the event loop.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start word prediciton GUI.', usage='\n* -n Number of word suggestions to display.')
    parser.add_argument('-n', type=int, default=20, help='Number of word suggestions to display.')
    parser.add_argument('--model-char-ngram', type=str, default='_fair_char_10gram.txt', help='Path to N-gram character model.')
    parser.add_argument('--model-char-neural', type=str, default='_neural_char_e20', help='Path to neural network character model.')
    parser.add_argument('--model-word-ngram', type=str, default='_fair_10gram.txt', help='Path to N-gram word model.')
    parser.add_argument('--model-word-neural', type=str, default='_lstm_e10', help='Path to neural network word model.')
    arguments = parser.parse_args()
    num_word_displayed = arguments.n
    path_char_ngram = arguments.model_char_ngram
    path_char_neural = arguments.model_char_neural
    path_word_ngram = arguments.model_word_ngram
    path_word_neural = arguments.model_word_neural

    probs_char_ngram = CharNGramProbabilities(path_char_ngram)
    probs_char_neural = CharNeuralProbabilities(path_char_neural)
    probs_word_ngram = NGramProbabilities(path_word_ngram)
    probs_word_neural = NeuralProbabilities(path_word_neural)
    window = Window(num_words_displayed=num_word_displayed)
    window.insert_word_probability('Character N-gram', probs_char_ngram)
    window.insert_word_probability('Character neural network', probs_char_neural)
    window.insert_word_probability('Word N-gram', probs_word_ngram)
    window.insert_word_probability('Word neural network', probs_word_neural)
    window.mainloop()
