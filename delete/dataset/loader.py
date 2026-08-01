import json

class GSM8KLoader:

    def __init__(self, path):
        with open(path) as f:
            self.data = json.load(f)

    def sample(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)