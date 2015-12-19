import json
import itertools

class SupertaggerConfig(object):

    def __init__(self, hyperparams):
        # Save as member variables for convenience.
        self.init_scale = hyperparams["init_scale"]
        self.seed = hyperparams["seed"]
        self.penultimate_hidden_size = hyperparams["penultimate_hidden_size"]
        self.num_layers = hyperparams["num_layers"]
        self.max_grad_norm = hyperparams["max_grad_norm"]
        self.regularize = hyperparams["regularize"]
        self.dropout_probability = hyperparams["dropout_probability"]

        shortened_hyperparams = { self.shorten(k):v for k,v in hyperparams.items() }
        if len(shortened_hyperparams) != len(hyperparams):
            raise ValueError("Shortened hyperparameter names not unique. Please rename them.")
        self.name = "-".join("{}_{}".format(k,v) for k,v in shortened_hyperparams.items())

    def shorten(self, name):
        return "".join(split[0] for split in name.split("_"))

def expand_grid(grid_file):
    # The grid is a json dictionary of lists of hyperparameters.
    with open(grid_file) as f:
        grid = json.load(f)
        return [SupertaggerConfig(dict(itertools.izip(grid, x))) for x in itertools.product(*(grid.itervalues()))]
