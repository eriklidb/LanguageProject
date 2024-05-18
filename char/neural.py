import string
import os
from data import Special, context_and_keystrokes
from trie import FreqTrie
import torch


class NeuralPredictor(torch.nn.Module):
    def __init__(self, data_src=None, max_ctx_len=3, epochs=None, device=None):
        super().__init__()
        if device is None and torch.cuda.is_available():
            device = 'cuda'
        elif device is None:
            device = 'cpu'
        self._device = device

        self._w2i = {}
        self._i2w = []
        self._trie = FreqTrie()

        self._NUM_SPECIAL_WORDS = Special.size()
        self.add_word(Special.PADDING)
        self.add_word(Special.UNKNOWN)
        self.add_word(Special.START)

        if data_src is not None:
            self.learn_vocab(data_src)
        self._init_params()
        if data_src is not None:
            if epochs is None:
                self.train_model(data_src)
            else:
                self.train_model(data_src, epochs=epochs)


    def train_model(self, data_src, epochs=1):
        # Training example from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html 
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters())
        running_loss = 0.0
        print('Starting training')
        self.train()
        batch_size = 16
        try:
            for epoch in range(epochs):
                print('epoch', epoch+1)
                epoch_loss = 0.0
                running_loss = 0.0
                for i, data in enumerate(data_src.labeled_samples_batch(batch_size)):
                    # get the inputs; data is a list of [inputs, labels]
                    contexts, labels = data #= NeuralPredictor.prep_sample(data)
                    labels = list(map(lambda l: Special.UNKNOWN if l not in self._w2i else l, labels))
                    labels = list(map(lambda l: self._w2i[l], labels))
                    labels = torch.tensor(labels).to(self._device)
                
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = self(contexts)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # print statistics
                    running_loss += loss.item()
                    epoch_loss += loss.item()
                    if i % 200 == 199:    # print every 200 mini-batches
                        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                        running_loss = 0.0
                print(f'[epoch {epoch + 1}] loss: {epoch_loss / (i+1):.3f}')
            self.eval()
            print('Finished Training')
        except KeyboardInterrupt:
            self.eval()
            print('Finished Training early')


    def _init_params(self):
        self._word_emb_size = 100
        self._vocab_size = len(self._w2i)
        self._word_emb = torch.nn.Embedding(self._vocab_size, self._word_emb_size)

         
        self._output_size = self._vocab_size
        word_hidden_size = 50
        rnn_layers = 1


        self._word_rnn = torch.nn.GRU(\
                self._word_emb_size,\
                word_hidden_size,\
                num_layers=rnn_layers)

        self._dropout = torch.nn.Dropout()

        layer_count = 0 
        layers = []
        for i in range(layer_count):
            layers.append(torch.nn.Linear(word_hidden_size, word_hidden_size))
            layers.append(torch.nn.Sigmoid())

        self._final = torch.nn.Sequential(*layers,\
                torch.nn.Linear(\
                rnn_layers * word_hidden_size,\
                self._output_size))

        self.to(self._device)

    
    def load(path, device=None):
        if device is None and torch.cuda.is_available():
            device = 'cuda'
        elif device is None:
            device = 'cpu'
        
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
        #sentence_state = sentence_state[0]
        batch_size = sentence_state.shape[1]
        sentence_state = sentence_state.reshape((batch_size, -1))
        sentence_state = self._dropout(sentence_state)
        logits = self._final(sentence_state)
        return logits


    def prep_sample(sample):
        text, label = sample
        context, keystrokes = context_and_keystrokes(text)
        return context, keystrokes, label


    def completions(self, context, keystrokes, n=1, deterministic=True):
        _, candidates, _ = self._trie.get_words(keystrokes)
        candidate_idx = list(map(lambda cand: self._w2i[cand], candidates))
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
    from data import DataSource, DataSourceNTComments
    data_path = 'data'
    if 'nyt' in data_path.lower():
        data_src = DataSourceNTComments(data_path, 300_000)
    else:
        data_src = DataSource(data_path, -1)
    print('Data source constructed')
    model = NeuralPredictor(data_src, 3, epochs=50)
    print("Save? (Y/n)")
    while True:
        ans = input()
        if ans == 'Y':
            print("Saving...")
            model.save('model_neural')
            print("Done saving.")
            break
        elif ans == 'n':
            break
            

