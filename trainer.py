from time import time

from tracker import Tracker
from model import Model

class Trainer():
    def train(self):
        m = Model()

        m.create()
        m.load_data()
        m.train()
        m.test()
        m.save_model()
