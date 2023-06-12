# Optimizing TAE training and improving ease of tensorizing different models

* `feature_test.ipynb` contains the basic code for the model and training, with sanity checks along the way to confirm operation.
* `code/train_utils.py` contains a few utility functions for creating progress bars, defining datasets, and training functions for simple AEs and the TAE. `train_utils_feature_demo.ipynb` illustrates the use of some of the features.
* `code/models.py` contains the definition of a base autoencoder class which can be used to define both linear and convolutional AEs.
* `code/tae.py` contains the definition of the TensorizedAutoencoder.
* `base.py` contains a few utility functions, not specific to the project
* 