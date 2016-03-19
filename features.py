import collections
import logging

import ccgbank

UNKNOWN_MARKER = "*UNKNOWN*"
OUT_OF_RANGE_MARKER = "*OOR*"

# Should define space and ispace.
class FeatureSpace(object):
    def index(self, f):
        return self.ispace[f]

    def feature(self, i):
        return self.space[i]

    def size(self):
        return len(self.space)

class SupertagSpace(FeatureSpace):
    def __init__(self, supertags_file):
        with open(supertags_file) as f:
            self.space = [line.strip() for line in f.readlines()]
            self.ispace = collections.defaultdict(lambda: -1, {f:i for i,f in enumerate(self.space)})

# Should define embedding_size.
class EmbeddingSpace(FeatureSpace):
    def extract(self, token):
        raise NotImplementedError("Subclasses must implement this!")

class TurianEmbeddingSpace(EmbeddingSpace):
    def __init__(self, embeddings_file):
        already_added = set()
        self.embedding_size = None
        self.space = []
        self.embeddings = []
        with open(embeddings_file) as f:
            for i,line in enumerate(f.readlines()):
                splits = line.split()
                word = splits[0]

                if i == 0 and word != UNKNOWN_MARKER:
                    raise ValueError("First embedding in the file should represent the unknown word.")

                word = word.lower()
                if word not in already_added:
                    embedding = [float(s) for s in splits[1:]]
                    if self.embedding_size is None:
                        self.embedding_size = len(embedding)
                    elif self.embedding_size != len(embedding):
                        raise ValueError("Dimensions mismatch. Expected {} but was {}.".format(self.embedding_size, len(embedding)))

                    already_added.add(word)
                    self.space.append(word)
                    self.embeddings.append(embedding)

        # Extra markers and tokens not found in the the embeddings file.
        self.space.append(ccgbank.START_MARKER)
        self.embeddings.append([0.0] * self.embedding_size)
        self.space.append(ccgbank.END_MARKER)
        self.embeddings.append([0.0] * self.embedding_size)
        self.space.append("")
        self.embeddings.append([0.0] * self.embedding_size)

        self.space = list(self.space)
        self.ispace = collections.defaultdict(lambda:0, {f:i for i,f in enumerate(self.space)})

    def extract(self, token):
        return token.lower()

class WordSpace(EmbeddingSpace):
    def extract(self, token):
        return token.lower()

class PrefixSpace(EmbeddingSpace):
    def __init__(self, n):
        self.n = n
        self.embedding_size = 32

    def extract(self, token):
        if token == ccgbank.START_MARKER or token == ccgbank.END_MARKER:
            return token
        else:
            return token[:self.n] if len(token) >= self.n else OUT_OF_RANGE_MARKER

class SuffixSpace(EmbeddingSpace):
    def __init__(self, n):
        self.n = n
        self.embedding_size = 32

    def extract(self, token):
        if token == ccgbank.START_MARKER or token == ccgbank.END_MARKER:
            return token
        else:
            return token[-self.n:] if len(token) >= self.n else OUT_OF_RANGE_MARKER

class EmpiricalEmbeddingSpace(EmbeddingSpace):
    def __init__(self, sentences, min_count):
        counts = collections.Counter()
        for tokens,supertags,weights in sentences():
            counts.update((self.extract(t) for t in tokens))

        self.space = [f for f in counts if counts[f] >= min_count]
        self.default_index = len(self.space)
        self.ispace = collections.defaultdict(lambda:self.default_index, {f:i for i,f in enumerate(self.space)})
        self.space.append(UNKNOWN_MARKER)

class EmpiricalPrefixSpace(EmpiricalEmbeddingSpace, PrefixSpace):
    def __init__(self, n, sentences, min_count=3):
        PrefixSpace.__init__(self, n)
        EmpiricalEmbeddingSpace.__init__(self, sentences, min_count)

class EmpiricalSuffixSpace(EmpiricalEmbeddingSpace, SuffixSpace):
    def __init__(self, n, sentences, min_count=3):
        SuffixSpace.__init__(self, n)
        EmpiricalEmbeddingSpace.__init__(self, sentences, min_count)
