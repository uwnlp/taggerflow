import util

START_MARKER = "<s>"
END_MARKER = "</s>"
TRITRAINING_WEIGHT = 0.3

class SupertagReader(object):
    def get_word_and_supertag(self, split):
        # Original CCGBank data.
        if len(split) == 3:
            return (split[0].strip(), split[2].strip(), 1.0)
        # Tritraining data.
        elif len(split) == 2:
            return (split[0].strip(), split[1].strip(), TRITRAINING_WEIGHT)
        else:
            raise ValueError("Unknown split length: {}".format(split))

    def get_sentences(self, filepath):
        with open(filepath) as f:
            sentences = [zip(*[self.get_word_and_supertag(split.split("|")) for split in line.split(" ")]) for line in f.readlines()]
            return [([START_MARKER] + list(words) + [END_MARKER], [None] + list(supertags) + [None], [0] + list(weights) + [0]) for words,supertags,weights in sentences]

    def get_splits(self, train_file):
        return [self.get_sentences(util.maybe_download("data",
                                                       "http://appositive.cs.washington.edu/resources/",
                                                       split_name + ".stagged")) for split_name in (train_file, "dev")]
