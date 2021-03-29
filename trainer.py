from time import time

from tracker import Tracker
from model import Model

class Trainer():
    def train(self):
        m = Model()

        m.create()
        m.load_data(['train-3.json', 'train-4.json', 'train-5.json'])
        m.train()
        m.test()
        m.save_model()
