import util
import itertools

START_MARKER = "<s>"
END_MARKER = "</s>"

class SupertagReader(object):
    def get_word_and_supertag(self, split):
        if len(split) == 3:
            return (split[0].strip(), split[2].strip())
        elif len(split) == 2:
            return (split[0].strip(), split[1].strip())
        else:
            raise ValueError("Unknown split length: {}".format(split))

    def get_sentences(self, filepath, is_tritrain, debug):
        with open(filepath) as f:
            lines = f.readlines()
            if is_tritrain and debug:
                lines = itertools.islice(lines, 10)
            sentences = [zip(*[self.get_word_and_supertag(split.split("|")) for split in line.split(" ")]) for line in lines]
            return [([START_MARKER] + list(words) + [END_MARKER],
                     [None] + list(supertags) + [None],
                     is_tritrain) for words,supertags in sentences]

    def get_split(self, split_name, is_tritrain, debug):
        return self.get_sentences(util.maybe_download("data",
                                                       "http://appositive.cs.washington.edu/resources/",
                                                       split_name + ".stagged"), is_tritrain, debug)

    def get_splits(self, debug=False):
        splits = [("train", False),
                  ("tritrain", True),
                  ("dev", False)]
        return [self.get_split(split_name, is_tritrain, debug) for split_name,is_tritrain in splits]
