import os
import re

from pyparsing import nums, printables
from pyparsing import Word, Forward, Group, OneOrMore, Literal

class CCGBankReader(object):

    TRAIN_REGEX = re.compile(r"wsj_((0[2-9])|(1[0-9])|(2[0-1])).*auto")
    DEV_REGEX = re.compile(r"wsj_00.*auto")
    TEST_REGEX = re.compile(r"wsj_23.*auto")

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

    def get_sentences(self, filename):
        with open(filename) as f:
            for line in f.readlines():
                if line.startswith("("):
                    yield zip(*self.ccgparse.parseString(line).asList())

    def get_splits(self, data_dir):
        train = []
        dev = []
        test = []
        for dirpath, dnames, fnames in os.walk(data_dir):
            print("Reading from {}".format(dirpath))
            for f in fnames:
                p = os.path.join(dirpath, f)
                if self.TRAIN_REGEX.match(f):
                    train.extend(self.get_sentences(p))
                elif self.DEV_REGEX.match(f):
                    dev.extend(self.get_sentences(p))
                elif self.TEST_REGEX.match(f):
                    test.extend(self.get_sentences(p))
        return (train, dev, test)
