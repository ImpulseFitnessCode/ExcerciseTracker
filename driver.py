import time
import numpy as np

from model import Model

class Driver():
    model = None
    tracker = None
    capture_pose = None
    # Reads per second
    resolution = 0.5

    window = [[] for _ in range(0, 12)]
    # rep window size in seconds
    window_size = 5 / resolution

    def __init__(self, tracker):
        self.model = Model()
        self.tracker = tracker
        self.capture_pose = tracker.capture_pose_gen()
        print('Creating driver...')
        self.model.load_model()


    def start(self):
        while True:
            data_piece = self.get_data_piece()
            self.add_to_window(data_piece)
            self.check_model()
            time.sleep(self.resolution)

    def check_model(self):
        if len(self.window[0]) >= self.window_size:
            pred = self.model.predict(self.window)
            self.print_prediction(pred)
    
    def print_prediction(self, prediction):
        pred_str = 'Overhead Press' if prediction[0] > 0.8 else 'None'
        print('Exercise: ' + pred_str)

    def add_to_window(self, item):
        for i, param in enumerate(self.window):
            if len(param) >= self.window_size:
                # Shift window
                param.pop(0)
            param.append(item[i])


    def get_data_piece(self):
        pose = next(self.capture_pose)[0]
        return np.array([ 
            pose[5],
            pose[6],
            pose[7],
            pose[8],
            pose[9],
            pose[10],
        ]).flatten()
