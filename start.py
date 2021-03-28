import tensorflow as tf
import sys
import argparse
sys.path.insert(1, './posenet-python')
import posenet

from trainer import Trainer
from tracker import Tracker
from driver import Driver
from test_data import TestData

tf.compat.v1.enable_eager_execution()

def create_test_data():
    tracker = Tracker()
    test_data = TestData(tracker)
    test_data.genTestData()
    test_data.processTestData()
    test_data.writeTestData()

def train():
    trainer = Trainer()
    trainer.train()

def run():
    tracker = Tracker()
    driver = Driver(tracker)
    driver.start()

def fetch_new_model():
    with tf.compat.v1.Session() as sess:
        posenet.load_model(101, sess, './posenet-python/_models')

parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('--fetch', action='store_const', const='fetch')
parser.add_argument('--test-data', action='store_const', const='test-data')
parser.add_argument('--train', action='store_const', const='train')
parser.add_argument('--run', action='store_const', const='run')

arg_options = {
    'fetch': fetch_new_model,
    'test-data': create_test_data,
    'train': train,
    'run': run,
}

opt = [val for name, val in vars(parser.parse_args()).items() if val is not None]
len(opt) and arg_options[opt[0]]()
