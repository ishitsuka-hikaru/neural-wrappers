import sys
import os
import h5py
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from argparse import ArgumentParser
from neural_wrappers.utilities import resize_black_bars
from tqdm import tqdm

def getArgs():
	parser = ArgumentParser()
	parser.add_argument("dir")
	parser.add_argument("--resolution", default="480,640")
	parser.add_argument("--out_file", default="dataset.h5")

	args = parser.parse_args()
	args.resolution = list(map(lambda x : int(x), args.resolution.split(",")))
	return args

def getPaths(baseDir):
	classes = filter(lambda x : x != "." and x != "..", os.listdir(baseDir))
	filePaths = {}
	for Class in classes:
		classDir = baseDir + os.sep + Class
		assert os.path.isdir(classDir)
		classFiles = filter(lambda x : os.path.isfile(x), map(lambda x : classDir + os.sep + x, os.listdir(classDir)))
		filePaths[Class] = list(classFiles)
		print("Class: %s. Num instances: %d" % (Class, len(filePaths[Class])))
	return filePaths

def readImage(path):
	tmp = np.array(Image.open(path), dtype=np.uint8)
	if len(tmp.shape) == 2:
		tmp = np.expand_dims(tmp, axis=-1)
	assert len(tmp.shape) == 3

	if tmp.shape[-1] == 4:
		img = tmp[..., 0 : 3]
	elif tmp.shape[-1] == 3:
		img = tmp
	elif tmp.shape[-1] == 1:
		img = np.zeros((*tmp.shape[0 : -1], 3), dtype=np.uint8)
		img[..., 0 : 3] = tmp
	else:
		assert False

	return img

def readAndStoreData(file, paths, resolution):
	tuplePaths = []
	for key in paths:
		for item in paths[key]:
			tuplePaths.append((item, key))
	random.shuffle(tuplePaths)

	file.create_group("train")
	file.create_group("validation")
	classes = list(paths.keys())
	file["classes"] = np.array(classes, "S")

	numFiles = len(tuplePaths)
	numTrain = int(0.8 * numFiles)
	numValidation = numFiles - numTrain
	file["train"].create_dataset("rgb", shape=(numTrain, resolution[0], resolution[1], 3), dtype=np.uint8)
	file["train"].create_dataset("label", shape=(numTrain, ), dtype=np.uint8)
	file["validation"].create_dataset("rgb", shape=(numValidation, resolution[0], resolution[1], 3), dtype=np.uint8)
	file["validation"].create_dataset("label", shape=(numValidation, ), dtype=np.uint8)

	for i in tqdm(range(numFiles)):
		path, pathClass = tuplePaths[i]
		img = readImage(path)
		imgClass = classes.index(pathClass)
		newImg = resize_black_bars(img, (resolution[0], resolution[1], 3))

		if i >= numTrain:
			ix = i - numTrain
			key = "validation"
		else:
			ix = i
			key = "train"

		file[key]["rgb"][ix] = newImg
		file[key]["label"][ix] = imgClass

def main():
	args = getArgs()
	baseDir = args.dir
	paths = getPaths(baseDir)

	file = h5py.File(args.out_file, "w")
	readAndStoreData(file, paths, args.resolution)
	file.flush()
	print("Successfully written to", args.out_file)

if __name__ == "__main__":
	main()