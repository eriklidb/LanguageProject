class FreqTrie:
    def __init__(self):
        self._root = FreqTrieNode()
    

    def add_word(self, word, n=1):
        self._root.add_suffix(word, n)


    def get_words(self, prefix):
        subroot = self._root.child(prefix, create=False)
        if subroot is None:
            return (0, [], [])

        nodes = subroot.descendants()
        total_freqs = subroot.subfreq()
        words = []
        freqs = []

        for node in nodes:
            if node.freq() > 0:
                words.append(node.word())
                freqs.append(node.freq())

        return (total_freqs, words, freqs)



class FreqTrieNode:
    def __init__(self, word=None):
        if word is None:
            self._word = ''
        else:
            self._word = word
        self._freq = 0
        self._subfreq = 0
        self._children = {}

    
    def add_suffix(self, suffix, n=1):
        if suffix == '':
            self.increment_freq(n)
        else:
            self.increment_subfreq(n)
            head = suffix[0]
            tail = suffix[1:]
            self.child(head).add_suffix(tail)
    

    def increment_freq(self, n=1):
        self._freq += n

    
    def increment_subfreq(self, n=1):
        self._subfreq += n


    def word(self):
        return self._word


    def freq(self):
        return self._freq


    def subfreq(self):
        return self._subfreq


    def child(self, suffix, create=True):
        if suffix == '':
            return self
        else:
            head = suffix[0]
            tail = suffix[1:]
            if head not in self._children:
                if not create:
                    return None
                child_word = self._word + head
                child = FreqTrieNode(child_word)
                self._children[head] = child
            return self._children[head].child(tail)


    def descendants(self):
        nodes = []
        for child in self._children.values():
            nodes.append(child)
            nodes.extend(child.descendants())
        return nodes
