import numpy as np


def remove_chapters(text):
    res = ''
    for i in text:
        try:
            int(i)
        except ValueError:
            res += i

    return res


def get_batches(array, batch_size, seq_len):
    # Using get_batches of udacity course
    batch_size_total = batch_size * seq_len
    n_batches = len(array) // batch_size_total

    array = array[:n_batches * batch_size_total].reshape((batch_size, -1))

    res = []
    for n in range(0, array.shape[1], seq_len):
        x = array[:, n:n + seq_len]
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], array[:, n + seq_len]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], array[:, 0]
        res.append([x, y])

    return res


class Dataset:

    def __init__(self, path, batch_size, seq_len, pre_proc=True):
        self.path = path
        self.batch_size = batch_size

        with open(self.path, 'r') as file:
            self.text = file.read()
        if pre_proc:
            self.text = remove_chapters(self.text)

        self.tokens = [x for x in self.text]
        self.chars = tuple(set(self.tokens))
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {char: num for num, char in self.int2char.items()}

        self.encoded = np.array([self.char2int[char] for char in self.tokens])

        self.batches = get_batches(self.encoded, batch_size, seq_len)

    def __getitem__(self, item):
        return self.batches


if __name__ == '__main__':
    data = Dataset('../data/lusiadas.txt', 8, 50)
    while(True):
        pass
