import os
import time
import urllib
import logging
import threading

def maybe_download(data_dir, source_url, filename):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        print("Using cached version of {}.".format(filepath))
    else:
        file_url = source_url + filename
        print("Downloading {}...".format(file_url))
        filepath, _ = urllib.urlretrieve(file_url, filepath)
        statinfo = os.stat(filepath)
        print("Succesfully downloaded {} ({} bytes).".format(file_url, statinfo.st_size))
    return filepath

class Timer:
    def __init__(self, name, active=True):
        self.name = name if active else None

    def __enter__(self):
        self.start = time.time()
        self.last_tick = self.start
        return self

    def __exit__(self, *args):
        if self.name is not None:
            logging.info("{} took {:.3f} seconds.".format(self.name, time.time() - self.start))

    def tick(self, message):
        current = time.time()
        logging.info("{} took {:.3f} seconds ({:.3f} seconds since last tick).".format(message, current - self.start, current - self.last_tick))
        self.last_tick = current

class ThreadedContext(object):
    def __init__(self):
          self.thread = threading.Thread(target=self.run)
          self.stop = False

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop = True
        self.thread.join()

    def run(self):
        while not self.stop:
            self.loop()

    def loop(self):
        raise NotImplementedError("Subclasses must implement this!")
