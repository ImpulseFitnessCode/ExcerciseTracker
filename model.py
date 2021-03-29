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
            keras.layers.Flatten(input_shape=(12, 10)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(2)
        ])
        self.m.compile(optimizer='adam',
            run_eagerly=True,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])


    def save_model(self, filename='model.tf'):
        path = self.model_dir + '/' + filename
        self.m.save(path, overwrite=True)

    def load_model(self, filename='model.tf'):
        print('Loading model...')
        path = self.model_dir + '/' + filename
        self.m = keras.models.load_model(path)
        self.prob_model = keras.Sequential([self.m, keras.layers.Softmax()])
        self.m.run_eagerly = True
        self.prob_model.run_eagerly = True
        self.prob_model.compile(run_eagerly=True)

    def train(self):
        self.m.fit(self.training_data, self.target_data, epochs=500, shuffle=True)

    def test(self):
        weights = self.m.get_weights()
        self.create()
        self.m.set_weights(weights)
        self.m.evaluate(self.test_data, self.test_target_data, verbose=2)

    def load_data(self, filenames=['train-1.json']):
        data_dir = 'training_data/'
        test_data_dir = 'test_data/'
        for filename in filenames:
            with open(data_dir + filename, 'r') as file:
                data = json.load(file)
                training_data, target_data = self.pre_process_data(data)
                self.training_data = np.concatenate((self.training_data,  training_data)) if self.training_data is not None else training_data
                self.target_data = np.concatenate((self.target_data,  target_data)) if self.target_data is not None else target_data
                print('Training data len: ', len(self.training_data))

        with open(test_data_dir + 'test_data.json', 'r') as file:
            data = json.load(file)
            test_data, test_target_data = self.pre_process_data(data)
            self.test_data = test_data
            self.test_target_data = test_target_data

    def pre_process_data(self, data):
        target_data = np.array(
            [item['expected'] for item in data]
        )
        data = np.array([
            [
                item['leftShoulderY'],
                item['leftShoulderX'],
                item['leftElbowY'],
                item['leftElbowX'],
                item['leftWristY'],
                item['leftWristX'],
                item['rightShoulderY'],
                item['rightShoulderX'],
                item['rightElbowY'],
                item['rightElbowX'],
                item['rightWristY'],
                item['rightWristX']
            ]
            for item in data
        ])
        return (data, target_data)

    def predict(self, data):
        predictions = self.prob_model.predict(tf.convert_to_tensor([data]))
        return [pred for pred in predictions][0]