from tkinter import Tk, Button


class Window(Tk):
    def __init__(self):
        super().__init__()

        self.title("Word Predictor")

        self.button = Button(text="Bottom text.")
        self.button.bind("<Button-1>", self.handle_button_press)
        self.button.pack()

    def handle_button_press(self, event):
        self.destroy()


# Start the event loop.
if __name__ == '__main__':
    window = Window()
    window.mainloop()
