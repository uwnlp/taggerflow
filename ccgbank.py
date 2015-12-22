import util

START_MARKER = "<s>"
END_MARKER = "</s>"

class SupertagReader(object):

    def get_sentences(self, filepath):
        with open(filepath) as f:
            sentences = [zip(*[(word.strip(),supertag.strip()) for word,pos,supertag in (split.split("|") for split in line.split(" "))]) for line in f.readlines()]
            return [([START_MARKER] + list(words) + [END_MARKER], [None] + list(supertags) + [None]) for words,supertags in sentences]

    def get_splits(self):
        return [self.get_sentences(util.maybe_download("data",
                                                       "http://appositive.cs.washington.edu/resources/",
                                                       split_name + ".stagged")) for split_name in ("train", "dev")]
