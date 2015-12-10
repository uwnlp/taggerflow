import os
import re
import urllib
import tarfile

from pyparsing import nums, printables
from pyparsing import Word, Forward, Group, OneOrMore, Literal

class CCGBankReader(object):
    TRAIN_REGEX = re.compile(r".*wsj_((0[2-9])|(1[0-9])|(2[0-1])).*auto")
    DEV_REGEX = re.compile(r".*wsj_00.*auto")
    TEST_REGEX = re.compile(r".*wsj_23.*auto")

    # For debugging purposes only.
    SMALL_TRAIN_REGEX = re.compile(r".*wsj_01.*auto")

    def __init__(self, supertags_only=True):
        self.ccgparse = Forward()
        supertag = Word(printables, excludeChars="<>")
        postag = Word(printables, excludeChars="<>")
        predarg = Word(printables, excludeChars="<>")
        token = Word(printables, excludeChars="<>")
        index = Word(nums)
        lbr = Literal("(").suppress()
        rbr = Literal(")").suppress()
        lbs = Literal("<").suppress()
        rbs = Literal(">").suppress()
        nonleaf = Group(lbs + Literal("T").suppress() + supertag + index + index + rbs) + OneOrMore(self.ccgparse)
        leaf = Group(lbs + Literal("L").suppress() + supertag + postag + postag + token + predarg + rbs)
        node = Group(leaf | nonleaf)
        self.ccgparse << Group(lbr + node + rbr)

        if supertags_only:
            # Flatten the parse.
            self.ccgparse.setParseAction(lambda s,l,t:t[0][0])
            nonleaf.setParseAction(lambda s,l,t:t.asList()[1:])

            # Only extract the tokens and supertags.
            leaf.setParseAction(lambda s,l,t:(t[0][3], t[0][0]))

    def get_sentences(self, tar, member):
        for line in tar.extractfile(member).readlines():
            if line.startswith("("):
                yield zip(*self.ccgparse.parseString(line).asList())

    def get_splits(self, debug=False):
        filepath = self.maybe_download("data",
                                       "http://appositive.cs.washington.edu/resources/",
                                       "LDC2005T13.tgz")
        print("Extracting data from {}...".format(filepath))
        train = []
        dev = []
        test = []
        train_regex = self.SMALL_TRAIN_REGEX if debug else self.TRAIN_REGEX
        with tarfile.open(filepath, "r:gz") as tar:
            for member in tar:
                if train_regex.match(member.name):
                    train.extend(self.get_sentences(tar, member))
                elif self.DEV_REGEX.match(member.name):
                    dev.extend(self.get_sentences(tar, member))
                elif self.TEST_REGEX.match(member.name):
                    test.extend(self.get_sentences(tar, member))
        return (train, dev, test)

    def maybe_download(self, data_dir, source_url, filename):
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
