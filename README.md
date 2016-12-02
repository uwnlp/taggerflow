# Taggerflow

This repository contains the code for training the supertagging model from [LSTM CCG Parsing](http://homes.cs.washington.edu/~kentonl/pub/llz-naacl.2016.pdf) at NAACL ([Lewis et al., 2016](http://homes.cs.washington.edu/~kentonl/pub/llz-naacl.2016.bib)).

## Dependencies
* Tensorflow (r0.11 or above)
 * https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#pip-installation

## Training and Evaluation
* `python taggerflow.py grid.json`
  * Trains a supertagging model.
  * Logs evaluation results.
  * Writes checkpoints to the log directory.

## Exporting to EasySRL
* `python taggerflow.py grid.json -c <checkpoint_path>`
  * Evaluates the checkpoint on the dev set as a sanity check.
  * Exports the model information to a temporary directory.
  * Prints the temporary directory with the exported model.
* The temporary directory should contain `graph.pb` and various `.txt` files.
* Download and extract http://lil.cs.washington.edu/resources/model_tritrain_finetune.tgz, which provides the correct file structure.
* Remove the existing `taggerflow` directory and replace it with the temporary directory.

## Running EasySRL
* Clone the EasySRL repository: https://github.com/uwnlp/EasySRL.
* Download http://lil.cs.washington.edu/resources/libtaggerflow.so and move it to the `lib` directory.
* EasySRL will use the trained supertagger for parsing when given the modified `model_tritrain_finetune` directory.
