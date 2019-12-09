import sys
import os
import numpy as np
import h5py
from mnist import MNIST

# The source_dir should contain the 4 items from http://yann.lecun.com/exdb/mnist/ (extracted, not .gz).

def main():
	assert len(sys.argv) == 3, "Usage: python mnist_converter.py source_dir output_path"
	sourceDir = os.path.abspath(sys.argv[1]) 
	outputPath = os.path.abspath(sys.argv[2])

	mnist_data = MNIST(sourceDir)
	trainData, trainLabels = mnist_data.load_training()
	testData, testLabels = mnist_data.load_testing()

	file = h5py.File(outputPath, "w")
	data_group = file.create_group("train")
	data_group = file.create_group("test")

	file["train"]["images"] = np.uint8(np.array(trainData).reshape((-1, 28, 28, 1)))
	file["train"]["labels"] = np.uint8(np.array(trainLabels))
	file["test"]["images"] = np.uint8(np.array(testData).reshape((-1, 28, 28, 1)))
	file["test"]["labels"] = np.uint8(np.array(testLabels))

	print("Done.")

if __name__ == "__main__":
	main()