import h5py
import numpy as np
import os
from neural_wrappers.utilities import tryReadImage, resize
from functools import partial
from argparse import ArgumentParser

'''
Final structure:
- train
    camera_2
      rgb
      labels
        label1
        ...
        labelN
      calibration
- test
    camera_2
      rgb
      calibration
'''

def getPaths(basePath):
	np.random.seed(42)
	paths = {}
	rgbFiles = sorted(os.listdir(basePath + os.sep + "image_2"))

	paths["label"] = np.array(list(map(lambda x : "%s/label_2/%s.txt" % (basePath, x[0 : -4]), rgbFiles)))
	paths["calib"] = np.array(list(map(lambda x : "%s/calib/%s.txt" % (basePath, x[0 : -4]), rgbFiles)))
	paths["rgb"] = np.array(list(map(lambda x : "%s/image_2/%s" % (basePath, x), rgbFiles)))
	perm = np.random.permutation(len(paths["rgb"]))
	for k in paths:
		paths[k] = paths[k][perm]

	# Now, split these dicts in train/val dicts.
	numData = len(paths["rgb"])
	numTrain = int(0.8 * numData)
	numVal = numData - numTrain

	print("Num data: %d. Num train: %d. Num validation: %d" % (numData, numTrain, numVal))
	newPathsDict = {"train" : {}, "validation" : {}}

	for k in paths:
		newPathsDict["train"][k] = paths[k][0 : numTrain]
	for k in paths:
		newPathsDict["validation"][k] = paths[k][numTrain : numTrain + numVal]

	return newPathsDict

def doPng(imagePath, resolution):
	return resize(tryReadImage(imagePath), height=resolution[0], width=resolution[1], interpolation="bilinear")

def doLabel(labelPath):
	classes = ["Pedestrian", "Truck", "Car", "Cyclist", "Misc", "Van", "Tram", "Person_sitting"]
	f = open(labelPath)
	lines = f.readlines()
	numLines = len(lines)
	labels = []

	for i in range(numLines):
		line = lines[i]
		object_label = line.split(" ")
		if not object_label[0] in classes:
			continue
		object_label[0] = classes.index(object_label[0])
		object_label[1 : ] = list(map(lambda x : float(x), object_label[1 : ]))
		labels.append(object_label)
	labels = np.array(labels, dtype=np.float32)
	f.close()
	# Labels are flattened at end, because they must be stored as 1D variable length array
	return labels.flatten()

def doCalibration(calibPath):
	f = open(calibPath)
	lines = f.readlines()
	for line in lines:
		if not line.startswith("P2"):
			continue
		items = line.strip().split(" ")[1 : ]
		calibration = np.asarray([float(number) for number in items]).reshape((3, 4))
	f.close()
	return calibration

def doDataset(file, paths, resolution):
	numData = {k : len(paths[k]["rgb"]) for k in paths}
	funcs = {
		"rgb" : partial(doPng, resolution=resolution),
		"calib" : doCalibration,
		"label" : doLabel
	}

	createDatasetArgs = {}
	for k in numData:
		createDatasetArgs[k] = {
			"rgb" : {"shape" : (numData[k], resolution[0], resolution[1], 3), "dtype" : np.float32},
			"calib" : {"shape" : (numData[k], 3, 4), "dtype" : np.float32},
			"label" : {"shape" : (numData[k], ), "dtype" : h5py.special_dtype(vlen=np.float32)}
		}

	for groupKey in paths:
		file.create_group(groupKey)

		for key in paths[groupKey]:
			shape, dtype = createDatasetArgs[groupKey][key]["shape"], createDatasetArgs[groupKey][key]["dtype"]
			file[groupKey].create_dataset(key, shape=shape, dtype=dtype)

		for i in range(numData[groupKey]):
			print("%s %d/%d done" % (groupKey, i + 1, numData[groupKey]), end="\r")
			for key in paths[groupKey]:
				item = funcs[key](paths[groupKey][key][i])
				file[groupKey][key][i] = item
		print("")

def getArgs():
	parser = ArgumentParser()
	parser.add_argument("base_directory")
	parser.add_argument("--rgb_resolution", default="375,1224")
	parser.add_argument("--output_file", default="dataset.h5")

	args = parser.parse_args()
	args.rgb_resolution = list(map(lambda x : float(x), args.rgb_resolution.split(",")))
	return args

def main():
	args = getArgs()
	paths = getPaths(args.base_directory)

	file = h5py.File(args.output_file, "w")
	doDataset(file, paths, resolution=args.rgb_resolution)

	# file = h5py.File("dataset.h5", "a")
	# baseDir = os.path.abspath(sys.argv[1])
	# doDataset(file, baseDir + os.sep + "training", "train", labelData=True, startIndex=int(sys.argv[2]))
	# doDataset(file, baseDir + os.sep + "testing", "test", labelData=False, startIndex=int(sys.argv[3]))

if __name__ == "__main__":
	main()