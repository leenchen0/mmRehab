import sys
import os

_logger = None

def log(*args, **kwargs):
    print(*args, file=_logger, **kwargs)


class Logger(object):
    def __init__(self, filename="default.log", overwrite=False):
        self.terminal = sys.stdout
        filepath, _ = os.path.split(filename)
        if not os.path.exists(filepath):
            try:
                os.makedirs(filepath)
            except Exception as e:
                print(f"An exception occurred: {e}")
        self.log = open(filename, 'w' if overwrite else 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.log.flush()


def init(logger):
    global _logger
    _logger = logger
