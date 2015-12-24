import logging
import collections
import re

import numpy as np

from features import *

class ParameterReader(object):
    def readline(self, line):
        raise NotImplementedError("Subclasses must implement this!")

    def get_result(self):
        raise NotImplementedError("Subclasses must implement this!")

class EmbeddingsReader(ParameterReader):
    embedding_regexes = {
        "prefix_(\d)" : lambda g: PrefixSpace(int(g[0])),
        "suffix_(\d)" : lambda g: SuffixSpace(int(g[0])),
        "words" : lambda g: WordSpace()
    }

    unknown_marker = "*UNKNOWN*"

    def __init__(self, name):
        self.name = name
        self.words = []
        self.embeddings = []
        self.default_index = None
        self.embedding_size = None

    def readline(self, i, line):
        splits = line.split()
        word = splits[0]

        if word == self.unknown_marker or word == self.unknown_marker.lower():
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
        embedding_space = None
        for regex, space_function in self.embedding_regexes.items():
            match = re.match(regex, self.name)
            if match:
                embedding_space = space_function(match.groups())
                break
        if embedding_space is None:
            raise ValueError("Unknown embedding space: {}".format(self.name))
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
        return np.array(self.matrix)

class Parameters:
    readers = {
        "EMBEDDINGS" : EmbeddingsReader,
        "PARAMETERS" : MatrixReader
    }

    param_header_regex = "\*(.*)\*(.*)"

    def __init__(self, embedding_spaces=[]):
        self.matrices = {}
        self.embedding_spaces = collections.OrderedDict(embedding_spaces)

    def read(self, filename):
        current_reader = None
        offset = 0
        with open(filename) as f:
            for i,line in enumerate(f.readlines()):
                line = line.strip()
                if current_reader is None:
                    param_type, name = re.match(self.param_header_regex, line).groups()
                    name = name.strip().replace(" ", "_").lower()
                    current_reader = self.readers[param_type](name)
                    offset = i + 1
                elif len(line) == 0:
                    if isinstance(current_reader, EmbeddingsReader):
                        self.embedding_spaces[current_reader.name] = current_reader.get_result()
                    elif isinstance(current_reader, MatrixReader):
                        self.matrices[current_reader.name] = current_reader.get_result()
                    else:
                        raise ValueError("Unknown reader type: {}".format(type(current_reader)))
                    current_reader = None
                else:
                    current_reader.readline(i - offset, line)

            logging.info("Loaded pretrained embedding spaces: {}".format(self.embedding_spaces.keys()))
            logging.info("Loaded pretrained matrices: {}".format(self.matrices.keys()))
