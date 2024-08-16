class Datagen:
    def __init__(self, data, batch_size):
        self.X = data[0]
        self.Y = data[1]
        self.batch_size = batch_size

    def __len__(self):
        return len(self.X) // self.batch_size

    def __iter__(self):
        start_idx = 0
        end_idx = start_idx + self.batch_size
        for idx in range(self.__len__()):
            yield self.X[start_idx:end_idx, :], self.Y[start_idx:end_idx, :]
            start_idx = end_idx
            end_idx = start_idx + self.batch_size
