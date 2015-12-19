import os
import time
import urllib
import logging
import threading
import datetime

def maybe_download(data_dir, source_url, filename):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        logging.info("Using cached version of {}.".format(filepath))
    else:
        file_url = source_url + filename
        logging.info("Downloading {}...".format(file_url))
        filepath, _ = urllib.urlretrieve(file_url, filepath)
        statinfo = os.stat(filepath)
        logging.info("Succesfully downloaded {} ({} bytes).".format(file_url, statinfo.st_size))
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
            logging.info("{} duration was {}.".format(self.name, self.readable(time.time() - self.start)))

    def readable(self, seconds):
        return str(datetime.timedelta(seconds=int(seconds)))

    def tick(self, message):
        current = time.time()
        logging.info("{} took {} ({} since last tick).".format(message, self.readable(current - self.start), self.readable(current - self.last_tick)))
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

class LoggingToFile(object):
    def __init__(self, logdir, filename):
        self.handler = logging.FileHandler(os.path.join(logdir, filename))

    def __enter__(self):
        logging.getLogger().addHandler(self.handler)

    def __exit__(self, *args):
        logging.getLogger().removeHandler(self.handler)
