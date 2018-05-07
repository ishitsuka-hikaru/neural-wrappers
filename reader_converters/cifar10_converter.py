import sys
import os
import numpy as np
import h5py
import pickle
from Mihlib import *

def unpickleFile(file):
	dataDict = pickle.load(file, encoding="bytes")
	data = np.uint8(dataDict[b"data"].reshape((-1, 3, 32, 32)).swapaxes(1, 2).swapaxes(2, 3))
	labels = np.uint8(np.array(dataDict[b"labels"]))
	return data, labels

def main():
	assert len(sys.argv) == 3, "Usage: python mnist_converter.py source_dir output_path"
	sourceDir = os.path.abspath(sys.argv[1])
	outputPath = os.path.abspath(sys.argv[2])

	trainFiles = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
	testFile = "test_batch"

	file = h5py.File(outputPath, "w")
	data_group = file.create_group("train")
	data_group = file.create_group("test")

	trainData = np.zeros((50000, 32, 32, 3), dtype=np.uint8)
	trainLabels = np.zeros((50000, ), dtype=np.uint8)

	for i, fileName in enumerate(trainFiles):
		path = sourceDir + os.sep + fileName
		f = open(path, "rb")
		data, labels = unpickleFile(f)
		trainData[i * 10000 : (i + 1) * 10000] = data
		trainLabels[i * 10000 : (i + 1) * 10000] = labels
		f.close()

	f = open(sourceDir + os.sep + testFile, "rb")
	testData, testLabels = unpickleFile(f)
	f.close()

	file["train"]["images"] = trainData
	file["train"]["labels"] = trainLabels
	file["test"]["images"] = testData
	file["test"]["labels"] = testLabels

if __name__ == "__main__":
	main()