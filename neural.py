import string
import os
from data import Special, clean, context_and_keystrokes
from trie import FreqTrie
import torch


class NeuralPredictor(torch.nn.Module):
    def __init__(self, data_src=None, max_ctx_len=3, epochs=None, device='cpu'):
        super().__init__()
        self._device = device

        self._w2i = {}
        self._i2w = []
        self._trie = FreqTrie()

        self._NUM_SPECIAL_WORDS = Special.size()
        self.add_word(Special.UNKNOWN)
        self.add_word(Special.PADDING)
        self.add_word(Special.START)
        CHARS = Special.all() + ['\''] + list(string.ascii_letters) + list(string.punctuation)
        self._c2i = {c:i for i,c in enumerate(CHARS)}
        self._char_count = len(CHARS)

        if data_src is not None:
            self.learn_vocab(data_src)
        self._init_params()
        if data_src is not None:
            if epochs is None:
                self.train_model(data_src, max_ctx_len)
            else:
                self.train_model(data_src, max_ctx_len, epochs=epochs)


    def train_model(self, data_src, max_ctx_len, epochs=1):
        # Training example from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html 
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        running_loss = 0.0
        print('Starting training')
        self.train()
        for epoch in range(epochs):
            print('epoch', epoch)
            epoch_loss = 0.0
            for i, data in enumerate(data_src.labeled_samples(1, max_ctx_len)):
                # get the inputs; data is a list of [inputs, labels]
                context, label_ = data #= NeuralPredictor.prep_sample(data)
                label = Special.UNKNOWN if label_ not in self._w2i else label_
                label = self._w2i[label]
                label -= self._NUM_SPECIAL_WORDS
                label = torch.tensor([[label]]).to(self._device)
                label = torch.nn.functional.one_hot(label, num_classes = self._vocab_size - self._NUM_SPECIAL_WORDS)\
                        .float()
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(context)
                pred = torch.argmax(outputs, dim=-1) + self._NUM_SPECIAL_WORDS
                #print(' ctx:', context)
                #print('labl:', label_)
                #print('pred:', self._i2w[pred[0][0]])
                loss = criterion(outputs[0], label[0])
                loss.backward()
                optimizer.step()
                
                # print statistics
                running_loss += loss.item()
                epoch_loss += loss.item()
                if i % 2000 == 1999:    # print every 200 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
            print(f'[epoch {epoch + 1}] loss: {epoch_loss / i:.3f}')
        self.eval()
        print('Finished Training')


    def _init_params(self):
        self._bidirectional = True
        self._char_emb_size = 10
        self._word_emb_size = 10
        if self._word_emb_size % 2 == 1 and self._bidirectional:
            self._word_emb_size += 1
        self._vocab_size = len(self._w2i)
        self._word_emb = torch.nn.Embedding(self._vocab_size, self._word_emb_size)
        self._char_emb = torch.nn.Embedding(self._char_count, self._char_emb_size)

        char_hidden_size = int(self._word_emb_size / 2) if self._bidirectional else self._word_emb_size
        word_hidden_size = 50
        
        self._char_rnn = torch.nn.GRU(\
                self._char_emb_size,\
                char_hidden_size,\
                bidirectional=self._bidirectional)
        
        self._word_rnn = torch.nn.GRU(\
                self._word_emb_size,\
                word_hidden_size)
        

        self._output_size = self._vocab_size - self._NUM_SPECIAL_WORDS
        self._final_layer = torch.nn.Linear(\
                word_hidden_size,\
                self._output_size)
        self._softmax = torch.nn.Softmax(dim=1)

        self.to(self._device)

    
    def load(path, device='cpu'):
        model = NeuralPredictor(device=device)
        
        state = os.path.join(path, 'state.pt')
        words = os.path.join(path, 'words.txt')
        for f in [state, words]:
            if not os.path.isfile(f):
                raise ValueError(f'Could not find model at path {path}')

        with open(words, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                word = line.strip()
                model.add_word(word)
        
        model._init_params()

        model.load_state_dict(torch.load(state, map_location=torch.device(device)))
        model.eval()
        return model


    def save(self, path):
        state = os.path.join(path, 'state.pt')
        words = os.path.join(path, 'words.txt')
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(self.state_dict(), state)

        with open(words, 'w', encoding='utf-8') as f:
            for word in self._i2w[self._NUM_SPECIAL_WORDS:]:
                f.writelines([f'{word}\n'])

    
    def add_word(self, word):
        if word not in self._w2i:
            idx = len(self._i2w)
            self._w2i[word] = idx
            self._i2w.append(word)
            if word not in Special.all():
                self._trie.add_word(word)

    def learn_vocab(self, data_src):
        for word in data_src.vocab():
            self.add_word(word)


    def forward(self, x):
        if type(x) == type(''):
            x = [x]
        x = map(lambda ctx: ctx.split(), x)
        x = list(x)
        x = map(lambda ctx: [Special.START] if len(ctx) == 0 else ctx, x)
        x = list(x)
        max_len = max(map(len, x))
        x = map(\
                lambda ctx: [Special.PADDING] * (max_len - len(ctx)) + ctx,\
                x)
        x = list(x)
        x = map(\
                lambda ctx: list(map(\
                lambda w: w if w in self._w2i else Special.UNKNOWN,\
                ctx)),\
                x)
        x = list(x)
        x = list(map(\
                lambda ctx:\
                list(map(lambda w: self._w2i[w], ctx)),\
                x))
        ctx = torch.tensor(x, dtype=torch.long).to(self._device)
        ctx = ctx.permute((1, 0))
        ctx_emb = self._word_emb(ctx).float()
        
        _, sentence_state = self._word_rnn(ctx_emb)

        logits = self._final_layer(sentence_state)
        return logits #self._softmax(logits)


    def prep_sample(sample):
        text, label = sample
        context, keystrokes = context_and_keystrokes(text)
        return context, keystrokes, label


    def completions(self, context, keystrokes, n=1, deterministic=True):
        _, candidates, _ = self._trie.get_words(keystrokes)
        candidate_idx = list(map(lambda cand: self._w2i[cand] - self._NUM_SPECIAL_WORDS, candidates))
        prob = torch.flatten(self(context))[candidate_idx]
        prob = torch.nn.functional.softmax(prob, dim=0)
        prob = prob.tolist()

        if n <= 0 or len(candidates) == 0:
            return [], []
        elif deterministic:
            det = list(zip(prob, candidates))
            det.sort(reverse=True)
            probs, completions = list(zip(*det[:n]))
            completions = list(completions)
            probs = list(probs)
            return completions, probs
        else:
            indices = np.random.choice(len(candidates),\
                    size=n,\
                    p=prob,\
                    replace=False)
            sampled = list(map(lambda i: (prob[i], candidates[i]), indices))
            sampled.sort(reverse=True)
            probs, completions = list(zip(*sampled[:n]))
            completions = list(completions)
            probs = list(probs)
            return completions, probs



if __name__ == '__main__':
    from data import DataSource
    data_path = 'data'
    data_src = DataSource(data_path)
    model = NeuralPredictor(data_src, 3, epochs=3)
    print("Save? (Y/n)")
    while True:
        ans = input()
        if ans == 'Y':
            print("Saving...")
            model.save('neural_model')
            print("Done saving.")
            break
        elif ans == 'n':
            break
            

