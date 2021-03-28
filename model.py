import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

class Model():
    m = None
    prob_model = None
    training_data = None
    target_data = None
    test_data = None
    test_target_data = None
    model_dir = './models'

    def create(self):
        self.m = keras.Sequential([
            keras.layers.Flatten(input_shape=(2, 25)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(2)
        ])
        self.m.compile(optimizer='adam',
            run_eagerly=True,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])


    def save_model(self, filename='model.tf'):
        path = self.model_dir + '/' + filename
        self.m.save(path)

    def load_model(self, filename='model.tf'):
        print('Loading model...')
        path = self.model_dir + '/' + filename
        self.m = keras.models.load_model(path)
        self.prob_model = keras.Sequential([self.m, keras.layers.Softmax()])
        self.m.run_eagerly = True
        self.prob_model.run_eagerly = True
        self.prob_model.compile(run_eagerly=True)

    def train(self):
        self.m.fit(self.training_data, self.target_data, epochs=1000)

    def test(self):
        weights = self.m.get_weights()
        self.create()
        self.m.set_weights(weights)
        self.m.evaluate(self.test_data, self.test_target_data, verbose=2)


    def load_data(self, data_dir='training_data/'):
        with open(data_dir + 'train.json', 'r') as file:
            data = json.load(file)
            training_data, target_data = self.pre_process_data(data)
            self.training_data = training_data
            self.target_data = target_data
            print('Training data len: ', len(self.training_data))

        with open(data_dir + 'test_data.json', 'r') as file:
            data = json.load(file)
            test_data, test_target_data = self.pre_process_data(data)
            self.test_data = test_data
            self.test_target_data = test_target_data

    def pre_process_data(self, data):
        target_data = np.array(
            [item['expected'] for item in data]
        )
        data = np.array([
            [item['nosePoseY'], item['nosePoseX']]
            for item in data
        ])
        return (data, target_data)

    def predict(self, data):
        predictions = self.prob_model.predict(tf.convert_to_tensor([data]))
        return [pred for pred in predictions]