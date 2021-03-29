from time import time

class PerfTest():
    def __init__(self, msg=None):
        self.msg = msg

    def __enter__(self):
        print('Performance test...')
        self.t1 = time()

    def __exit__(self, exc_type, exc_value, traceback):
        print(('{0}: '.format(self.msg) if self.msg else 'Time: ') + str(time() - self.t1) + ' Seconds')
