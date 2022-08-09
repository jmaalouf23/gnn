# ylide_gnn

Machine learning models to predict the properties of redox mediators.

This repository uses chemprop's development of a message passing neural network (https://github.com/chemprop/chemprop) to predict thermodynamic properties of ylide redox mediators. The Chemprop MPN class is wrapped in custom code to add a mlp readout layer after the molecule undergoes featurization and message passing to become vectorized.

The current implementation performs random splits on the data.

Currently, the models have been trained to predict properties that are important for redox mediator performance including redox potential, deprotonation free energy, and Hydrogen abstraction energy.


Models can be trained by running python train.py.


