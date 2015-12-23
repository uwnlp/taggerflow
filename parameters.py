import logging
import collections

from features import *

UNKNOWN_MARKER = "*UNKNOWN*"

class ParameterReader(object):
    def readline(self, line):
        raise NotImplementedError("Subclasses must implement this!")

    def get_result(self):
        raise NotImplementedError("Subclasses must implement this!")

class EmbeddingsReader(ParameterReader):
    def __init__(self, name):
        self.name = name
        self.words = []
        self.embeddings = []
        self.default_index = None
        self.embedding_size = None

    def readline(self, i, line):
        splits = line.split()
        word = splits[0]

        if word == UNKNOWN_MARKER or word == UNKNOWN_MARKER.lower():
            if self.default_index is None:
                self.default_index = i
            else:
                raise ValueError("Unknown word repeated.")

        embedding = [float(s) for s in splits[1:]]
        if self.embedding_size is None:
            self.embedding_size = len(embedding)
        elif self.embedding_size != len(embedding):
            if self.embedding_size == len(embedding) + 1:
                # Assume this corresponds to the empty string.
                word = ""
                embedding = [float(s) for s in splits]
            else:
                raise ValueError("Dimensions mismatch. Expected {} but was {}.".format(self.embedding_size, len(embedding)))

        self.words.append(word)
        self.embeddings.append(embedding)

    def get_result(self):
        if self.default_index is None:
            return ValueError("Unknown word not found.")
        embedding_space = PretrainedEmbeddingSpace()
        embedding_space.embedding_size = self.embedding_size
        embedding_space.space = self.words
        embedding_space.ispace = collections.defaultdict(lambda:self.default_index, {f:i for i,f in enumerate(self.words)})
        embedding_space.embeddings = self.embeddings
        return embedding_space

class MatrixReader(ParameterReader):
    def __init__(self, name):
        self.name = name
        self.matrix = []
        self.dimensions = None

    def readline(self, i, line):
        if self.dimensions is None:
            # {columns,rows} or {rows}
            line = line[line.index("{") + 1:line.index("}")]
            self.dimensions = [int(s) for s in line.split(",")]
            if len(self.dimensions) != 1 and len(self.dimensions) != 2:
                raise ValueError("Unsupported shape: {}".format(self.dimensions))
        else:
            splits = line.split()
            expected_column_size = 1 if len(self.dimensions) == 1 else self.dimensions[1]
            if len(splits) != expected_column_size:
                raise ValueError("Expected column size {} but was {}".format(expected_column_size, len(splits)))
            self.matrix.append([float(s) for s in splits])

    def get_result(self):
        if len(self.matrix) != self.dimensions[0]:
            raise ValueError("Expected row size {} but was {}.".format(self.dimensions[0], len(self.matrix)))
        return self.matrix

readers = {
    "EMBEDDINGS" : EmbeddingsReader,
    "PARAMETERS" : MatrixReader
}

def read(filename):
    current_reader = None
    offset = 0
    params = {}
    with open(filename) as f:
        for i,line in enumerate(f.readlines()):
            line = line.strip()
            if current_reader is None:
                name_start = line.index("*") + 1
                name_end = line.index("*", name_start)
                current_reader = readers[line[name_start:name_end]](line[name_end + 1:].strip())
                offset = i + 1
            elif len(line) == 0:
                params[current_reader.name] = current_reader
                current_reader = None
            else:
                current_reader.readline(i - offset, line)
    return {k:v.get_result() for k,v in params.items()}

if __name__ == "__main__":
    params = read("data/parameters")
    print("Params: {}".format(params.keys()))
