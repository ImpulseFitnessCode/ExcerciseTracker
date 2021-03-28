import time

from model import Model

class Driver():
    model = None
    tracker = None
    capture_pose = None
    # Reads per second
    resolution = 0.2

    window = [[] for _ in range(0, 2)]
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
            print(pred)

    def add_to_window(self, item):
        for i, param in enumerate(self.window):
            if len(param) >= self.window_size:
                # Shift window
                param.pop(0)
            param.append(item[i])


    def get_data_piece(self):
        return next(self.capture_pose)[0][0]
