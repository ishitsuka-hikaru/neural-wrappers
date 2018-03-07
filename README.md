How to install:
- Add the path to the src directory of this project in PYTHONPATH environment variable.
    - Example .bashrc: `export PYTHONPATH="$PYTHONPATH:/path/to/neural-wrappers/src"`

Structure of this project:
- examples/ 
    - mnist-classifier - Simple MNIST classifier using 2 networks (FC and Conv). See this for basic usage of the framework
    - char-rnn - Implementation of a simple recurrent network that predicts one character after another. Inspired by: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
    - video-autoencoder - Implementation of a simple video auto-encoder, that takes two videos (say 240p and 120p) and creates a convolutional + LSTM network that tries to compress the given video. Can be used to generate a new video of a smaller dimension (or even increase resolution, if a better model would be implemented)
- test/
    - Unit tests for all the implemented modules (WIP)
    - To run tests, go in the tests directory and type 'pytest' in the console. Requires the pytest module to be
	 installed, which can be done using `pip install pytest`.
- src/
	- dataset_reader/ - Base class for all readers and various readers for known datasets.
	- models/ - Various models from different papers implemented. May contain additional or missing features from
		original articles (just PyTorch for now).
	- transforms/ - Basic transforms and some built-in transforms for data augmentation (mirror, cropping)
	- wrappers/ - Main wrapper directory
	    - pytorch/ - Files that implement various features on top of PyTorch framework
	- callbacks.py - Basic callback class and some built-in callbacks for training (history.txt, model saving)
	- metrics.py - Basic metrics and some built-in metrics for training (accuracy/loss)
	- utils.py - Various functions
- README.md - this file
