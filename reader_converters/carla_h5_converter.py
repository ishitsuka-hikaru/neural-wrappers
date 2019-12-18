# Converter for CARLA simulator. Exports to H5 file based on a given path to an Unreal/Carla export (of pngs).
import numpy as np
import os
import sys
import h5py
import matplotlib.pyplot as plt
from PIL import Image
from neural_wrappers.utilities import h5StoreDict

def getPaths(baseDir):
	def positionFunc(rgbItem):
		splits = rgbItem.split("_")
		x, y, z, roll, pitch, yaw = splits[5], splits[7], splits[9], splits[15][0 : -4], splits[11], splits[13]
		return float(x), float(y), float(z), float(roll), float(pitch), float(yaw)

	def idsFunc(rgbItem):
		id = int(rgbItem.split("_")[3])
		return id

	def depthFunc(rgbItem):
		return baseDir + os.sep + rgbItem.replace("rgb", "depth")

	def semanticFunc(rgbItem):
		return baseDir + os.sep + rgbItem.replace("rgb", "semantic_segmentation")

	def normalFunc(rgbItem):
		return baseDir + os.sep + rgbItem.replace("rgb", "normal")

	rgbList = sorted(list(filter(lambda x : x.find("rgb") != -1, os.listdir(baseDir))))
	N = len(rgbList)
	result = {
		"rgb" : list(map(lambda x : baseDir + os.sep + x, rgbList)),
		"depth" : list(map(depthFunc, rgbList)),
		"position" : list(map(positionFunc, rgbList)),
		"ids" : list(map(idsFunc, rgbList)),
		"semantic_segmentation" : list(map(semanticFunc, rgbList)),
		"normal" : list(map(normalFunc, rgbList))
	}

	# Sort entries by IDs
	argSort = np.argsort(np.array(result["ids"], dtype=np.int64))
	result = {k : np.array(result[k])[argSort] for k in result}
	# Remove duplicate entries
	sortedPos = result["position"]
	right = np.append(sortedPos[1 :], [[0, 0, 0, 0, 0, 0]], axis=0)
	mask = np.abs(sortedPos - right).sum(axis=-1) > 0.1
	print("Removed %d duplicate entries" % (len(mask) - mask.sum()))
	result = {k : result[k][mask] for k in result}
	return result

def getTrainValPaths(paths, keepN=None):
	np.random.seed(42)
	perm = np.random.permutation(len(paths["rgb"]))
	numTrain = int(len(paths["rgb"]) * 0.8)
	numVal = len(paths["rgb"]) - numTrain

	allIx = np.random.permutation(len(paths["rgb"]))
	# Optional: sort them ascending in array. I think this is a bad idea, as we can find out neighbours by searhing
	#  at startup time for closest neighbours (via distance or ids)
	trainIx = allIx[0 : numTrain]
	valIx = allIx[numTrain : ]

	trainPaths = {k : paths[k][trainIx] for k in paths}
	valPaths = {k : paths[k][valIx] for k in paths}

	if not keepN is None:
		trainPaths = {k : trainPaths[k][0 : keepN] for k in trainPaths}
		valPaths = {k : valPaths[k][0 : keepN] for k in valPaths}

	return trainPaths, valPaths

def storeToH5File(file, data):
	def doPng(path):
		i = 0
		while True:
			if i == 5:
				return np.zeros((854, 854, 3), dtype=np.uint8)
			try:
				img = Image.open(path)
				npImg = np.array(img, np.uint8)[..., 0 : 3]
				break
			except Exception as e:
				i += 1
		return npImg

	def doDepth(path):
		i = 0
		while True:
			if i == 5:
				return np.zeros((854, 854), dtype=np.float32)
			try:
				dph = np.array(Image.open(path), np.float32)[..., 0 : 3]
				dphNorm = (dph[..., 0] + dph[..., 1] * 256 + dph[..., 2] * 256 * 256) / (256 * 256 * 256 - 1) * 1000
				dphNormImg = Image.fromarray(dphNorm)
				npDphNorm = np.array(dphNormImg, dtype=np.float32)
				break
			except Exception as e:
				i += 1
		return npDphNorm

	def doSemantic(path):
		item = doPng(path)
		labels = {
			(0, 0, 0): "Unlabeled",
			(70, 70, 70): "Building",
			(153, 153, 190): "Fence",
			(160, 170, 250): "Other",
			(60, 20, 220): "Pedestrian",
			(153, 153, 153): "Pole",
			(50, 234, 157): "Road line",
			(128, 64, 128): "Road",
			(232, 35, 244): "Sidewalk",
			(35, 142, 107): "Vegetation",
			(142, 0, 0): "Car",
			(156, 102, 102): "Wall",
			(0, 220, 220): "Traffic sign"
		}
		labelKeys = list(labels.keys())
		result = np.zeros(shape=item.shape[0] * item.shape[1], dtype=np.uint8)
		flattenedRGB = item.reshape(-1, 3)

		for i, label in enumerate(labelKeys):
			equalOnAllDims = np.prod(flattenedRGB == label, axis=-1)
			where = np.where(equalOnAllDims == 1)[0]
			result[where] = i

		result = result.reshape(*item.shape[0 : 2])
		return result

	# Normals are stored as [0 - 255] on 3 channels, representing the normals w.r.t world. We move them to [-1 : 1]
	def doNormal(path):
		item = doPng(path)
		return ((np.float32(item) / 255) - 0.5) * 2

	N = len(data["rgb"])
	funcs = {
		"rgb" : doPng,
		"depth" : doDepth,
		"position" : lambda x : x,
		"ids" : lambda x : x,
		"semantic_segmentation" : doSemantic,
		"normal" : doNormal
	}

	# Infer the shape and dtype from first item
	for key in data:
		assert key in funcs, "Not found %s in funcs %s" % (key, list(funcs))
		item = funcs[key](data[key][0])
		file.create_dataset(key, (N, *item.shape), dtype=item.dtype)
		file[key][0] = item

	# Do the rest N-1 items identically
	for i in range(1, N):
		print("%d/%d" % (i + 1, N), end="\r")
		for key in data:
			file[key][i] = funcs[key](data[key][i])
	print("")

def plotPaths(trainPaths, valPaths):
	plt.scatter(trainPaths["position"][..., 0], trainPaths["position"][..., 1])
	plt.savefig("train_points.png")
	plt.figure()
	plt.scatter(valPaths["position"][..., 0], valPaths["position"][..., 1])
	plt.savefig("val_points.png")

def getDataStatistics(file, maxDepthMeters=300):
	statistics = {}
	positions = {k : file[k]["position"][0 : ] for k in ["train", "validation"]}
	posConcat = np.concatenate([positions[k] for k in positions])
	statistics["position"] = {"min" : posConcat.min(axis=0), "max" : posConcat.max(axis=0)}
	statistics["depth"] = {"min" : 0, "max" : maxDepthMeters}
	return {"dataStatistics" : statistics}

def main():
	assert len(sys.argv) == 3, "Usage: python3 h5_exporter.py baseDir result.h5"
	paths = getPaths(sys.argv[1])
	print("Got %d paths. Keys: %s" % (len(paths["rgb"]), list(paths.keys())))

	trainPaths, valPaths = getTrainValPaths(paths, keepN=20)
	print("Got paths: Train: %d; Val: %d" % (len(trainPaths["rgb"]), len(valPaths["rgb"])))
	plotPaths(trainPaths, valPaths)

	file = h5py.File(sys.argv[2], "w")
	file.create_group("train")
	print("Storing train set")
	storeToH5File(file["train"], trainPaths)

	file.create_group("validation")
	print("Storing validation set")
	storeToH5File(file["validation"], valPaths)

	# file = h5py.File(sys.argv[2], "r")
	print("Storing statistics!")
	statistics = getDataStatistics(file, maxDepthMeters=300)
	file.create_group("others")
	h5StoreDict(file["others"], statistics)

	file.flush()
	print("Done! Exported to %s." % (sys.argv[2]))

if __name__ == "__main__":
	main()
