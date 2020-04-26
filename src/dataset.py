class Dataset:

    def __init__(self, path, batch_size, one_hot=True):
        self.path = path
        self.batch_size = batch_size
        self.one_hot = one_hot
        with open(self.path, 'r') as file:
            self.text = file.read()


if __name__ == '__main__':
    data = Dataset('../data/lusiadas.txt', 1)
    print(data.text)