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

    def get_sentences(self, filepath, is_tritrain):
        with open(filepath) as f:
            lines = f.readlines()
            sentences = (itertools.izip(*[self.get_word_and_supertag(split.split("|")) for split in line.split(" ")]) for line in lines)
            return [([START_MARKER] + list(words) + [END_MARKER],
                     [None] + list(supertags) + [None],
                     is_tritrain) for words,supertags in sentences]

    def get_split(self, split_name, is_tritrain):
        return self.get_sentences(util.maybe_download("data",
                                                       "http://appositive.cs.washington.edu/resources/",
                                                       split_name + ".stagged"), is_tritrain)

    def get_splits(self, read_tritrain=True):
        train = self.get_split("train", is_tritrain=False)
        tritrain = self.get_split("tritrain", is_tritrain=True) if read_tritrain else []
        dev = self.get_split("dev", is_tritrain=False)
        return train, tritrain, dev
