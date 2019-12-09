How to install:
- Add the path to the root directory of this project in PYTHONPATH environment variable.
    - Example .bashrc: `export PYTHONPATH="$PYTHONPATH:/path/to/neural-wrappers/"`

Structure of this project:
- examples/
	- tutorials/
		- Basic tutorials on how to use the library
	- Some examples on how to use the library
	- TODO - more detail
- neural_wrappers/
	- TODO - more detail
- reader_converters/
	- Various converters from the form the dataset is offered online (usually a big archive of images and labels, textual or not) to the form that is accepted by an implemented reader (under neural_wrappers/readers), which uses the DatasetReader class API (compatible with Keras fit_generator method and NeuralNetworkPyTorch train_generator). Generally, these converters will generate one h5py file that is used in the reader.
- test/
    - Unit tests for all the implemented modules (WIP)
    - To run tests, go in the tests directory and type 'pytest' in the console. Requires the pytest module to be
	 installed, which can be done using `pip install pytest`
- README.md - this file
