import os
import re
import urllib
import tarfile

from pyparsing import nums, printables
from pyparsing import Word, Forward, Group, OneOrMore, Literal

import util

class CCGBankReader(object):
    TRAIN_REGEX = re.compile(r".*wsj_((0[2-9])|(1[0-9])|(2[0-1])).*auto")
    DEV_REGEX = re.compile(r".*wsj_00.*auto")
    TEST_REGEX = re.compile(r".*wsj_23.*auto")

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

    def get_splits(self):
        filepath = util.maybe_download("data",
                                       "http://appositive.cs.washington.edu/resources/",
                                       "LDC2005T13.tgz")
        print("Extracting data from {}...".format(filepath))
        train = []
        dev = []
        test = []
        with tarfile.open(filepath, "r:gz") as tar:
            for member in tar:
                if self.TRAIN_REGEX.match(member.name):
                    train.extend(self.get_sentences(tar, member))
                elif self.DEV_REGEX.match(member.name):
                    dev.extend(self.get_sentences(tar, member))
                elif self.TEST_REGEX.match(member.name):
                    test.extend(self.get_sentences(tar, member))
        return (train, dev, test)

class SupertagReader(object):

    def get_sentences(self, filepath):
        with open(filepath) as f:
            return [zip(*[(word.strip(),supertag.strip()) for word,pos,supertag in (split.split("|") for split in line.split(" "))]) for line in f.readlines()]

    def get_splits(self):
        return [self.get_sentences(util.maybe_download("data",
                                                       "http://appositive.cs.washington.edu/resources/",
                                                       split_name + ".stagged")) for split_name in ("train", "dev", "test")]
