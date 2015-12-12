import os
import urllib

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
