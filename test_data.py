import json
import time

from utils import PerfTest

class TestData():
    test_data = []
    # Seconds
    time_frame = 20
    # Time it takes to capture a frame
    capture_time = 0.27
    # Reads per second
    resolution = 1
    rep_interval = 5
    rep_length = 2.5

    processed_data = []
    data_size = 0
    # Rep window size in seconds
    window_size = 5

    def __init__(self, tracker, time_frame = None):
        self.capture_pose = tracker.capture_pose_gen()
        self.time_frame = time_frame if time_frame else self.time_frame
        self.data_size = (int)(self.time_frame / self.resolution)


    def genTestData(self):
        rep_started = False
        rep_start_time = 0
        rep_interval = (int)(self.rep_interval / self.resolution)
        rep_length = (int)(self.rep_length / self.resolution)
        rep_count = 0

        for i in range(0, self.data_size):
            if rep_started and i == rep_start_time + (rep_length / 2):
                print('%d / %d' % (rep_count, (int)(self.data_size / rep_interval) - 1))
            if i != 0 and i % rep_interval == 0:
                print('Start Rep')
                rep_count += 1
                rep_started = True
                rep_start_time = i
            if rep_started and rep_start_time + rep_length < i:
                print('Stop Rep')
                rep_started = False

            data_piece = self.getDataPiece()
            data_piece['is_rep'] = int(rep_started)
            self.test_data.append(data_piece)

            time.sleep(self.resolution - self.capture_time)


    def getDataPiece(self):
        return {
            'nosePose': next(self.capture_pose)[0][0]
        }



    def processTestData(self):
        print('Post processing data...')
        t1 = time.time()
        buffer_size = (int)(self.window_size / self.resolution)
        for i in range(buffer_size, self.data_size):
            window = self.test_data[i - buffer_size:i]
            expected = False
            rep_started = False
            for i, reading in enumerate(window[:-1]):
                prev = window[i]['is_rep']
                next = window[i+1]['is_rep']
                if prev == 0 and next == 1:
                    rep_started = True
                if rep_started and prev == 1 and next == 0:
                    expected = True

            processed_window = {
                'expected': int(expected),
                'nosePoseY': [read['nosePose'][0] for read in window],
                'nosePoseX': [read['nosePose'][1] for read in window],
            }
            self.processed_data.append(processed_window)

        print(self.processed_data[10])
        print(len(self.processed_data))
        print('Processing took: ' + str(time.time() - t1) + ' Seconds')


    def writeTestData(self, filename='training_data/train.json'):
        with open(filename, 'w') as file:
            json.dump(self.processed_data, file)
        print('Data written to train.json')
