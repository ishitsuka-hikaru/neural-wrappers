# Converter for CARLA simulator. Exports to H5 file based on a given path to an Unreal/Carla export (of pngs).
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
from PIL import Image
from neural_wrappers.utilities import h5StoreDict, h5ReadDict
from neural_wrappers.readers.carla_h5_reader import CarlaH5PathsReader
from argparse import ArgumentParser

def getArgs():
	parser = ArgumentParser()
	parser.add_argument("baseDir")
	parser.add_argument("resultFile")
	parser.add_argument("--N", type=int, default=None)
	parser.add_argument("--test_export", default=0, type=int)
	parser.add_argument("--splits", default="80,20")
	parser.add_argument("--split_keys", default="train,validation")
	parser.add_argument("--statistics_file")

	args = parser.parse_args()
	args.test_export = bool(args.test_export)
	if args.test_export:
		print("Test export. Setting --randomize_order=False, --split_keys=test and --splits=1")
		args.randomize_order = False
		args.split_keys = ["test"]
		args.splits = [1]
		assert not args.statistics_file is None, \
			"For test export, we need the path to the train set, so we can copy its statistics"
	else:
		args.randomize_order = True
		args.split_keys = args.split_keys.split(",")
		args.splits = list(map(lambda x : float(x) / 100, args.splits.split(",")))

	assert abs(sum(args.splits) - 1) < 1e-5
	assert len(args.splits) == len(args.split_keys)
	return args

def getPaths(baseDir):
	def positionFunc(rgbItem):
		splits = rgbItem.split("_")
		x, y, z, roll, pitch, yaw = splits[5], splits[7], splits[9], splits[15][0 : -4], splits[11], splits[13]
		return float(x), float(y), float(z), float(roll), float(pitch), float(yaw)

	def idsFunc(rgbItem):
		id = int(rgbItem.split("_")[3])
		return id

	def depthFunc(rgbItem):
		return rgbItem.replace("rgb", "depth")

	def semanticFunc(rgbItem):
		return rgbItem.replace("rgb", "semantic_segmentation")

	def normalFunc(rgbItem):
		return rgbItem.replace("rgb", "normal")

	rgbList = sorted(list(filter(lambda x : x.find("rgb") != -1, os.listdir(baseDir))))
	N = len(rgbList)
	result = {
		"rgb" : np.array(rgbList, "S"),
		"depth" : np.array(list(map(depthFunc, rgbList)), "S"),
		"position" : np.array(list(map(positionFunc, rgbList)), np.float32),
		"ids" : np.array(list(map(idsFunc, rgbList)), np.uint64),
		"semantic_segmentation" : np.array(list(map(semanticFunc, rgbList)), "S"),
		"normal" : np.array(list(map(normalFunc, rgbList)), "S")
	}

	# Sort entries by IDs
	argSort = np.argsort(result["ids"])
	result = {k : result[k][argSort] for k in result}
	# Remove duplicate entries
	sortedPos = result["position"]
	right = np.append(sortedPos[1 :], [[0, 0, 0, 0, 0, 0]], axis=0)
	mask = np.abs(sortedPos - right).sum(axis=-1) > 0.1
	print("Removed %d duplicate entries" % (len(mask) - mask.sum()))
	result = {k : result[k][mask] for k in result}
	return result

def getTrainValPaths(paths, splits, splitKeys, keepN=None, randomizeOrder=True):
	N = len(paths["rgb"])
	# rgb, depth, semantic etc.
	pathKeys = paths.keys()
	
	# Randomize order
	if randomizeOrder:
		np.random.seed(42)
		perm = np.random.permutation(N)
		paths = {k : paths[k][perm] for k in pathKeys}

	# Get (startIndex, endIndex) tuple for each key
	dataIx, lastIx = {}, 0
	for i in range(len(splitKeys) - 1):
		nK = int(splits[i] * N)
		dataIx[splitKeys[i]] = (lastIx, lastIx + nK)
		lastIx += nK
	dataIx[splitKeys[-1]] = (lastIx, N)

	# Now, take the paths for all the keys as defined by indexes above
	newPaths = {}
	for k in splitKeys:
		startIx, endIx = dataIx[k]
		thisPaths = {pathKey : paths[pathKey][startIx : endIx] for pathKey in pathKeys}
		newPaths[k] = thisPaths
		print(k)
		for pathKey in pathKeys:
			print(" - %s : %d" % (pathKey, len(thisPaths[pathKey])))

	if not keepN is None:
		for k in splitKeys:
			newPaths[k] = {pathKey : newPaths[k][pathKey][0 : keepN] for pathKey in pathKeys}

	for k in splitKeys:
		thisPaths = newPaths[k]
		print(k, "=>", end="")
		for pathKey in pathKeys:
			print(" %s : %d |" % (pathKey, len(thisPaths[pathKey])), end="")
		print("")
	return newPaths

def plotPaths(paths):
	for k in paths:
		plt.gcf().clear()
		plt.scatter(paths[k]["position"][..., 0], paths[k]["position"][..., 1])
		print("Storing PNG paths: %s_points.png" % (k))
		plt.savefig("%s_points.png" % (k))

def storeToH5File(baseDir, file, data):
	N = len(data["rgb"])
	funcs = {
		"rgb" : CarlaH5PathsReader.doPng,
		"depth" : CarlaH5PathsReader.doDepth,
		"position" : lambda x, _ : x,
		"ids" : lambda x, _ : x,
		"semantic_segmentation" : CarlaH5PathsReader.doSemantic,
		"normal" : CarlaH5PathsReader.doNormal
	}

	# Infer the shape and dtype from first item
	for key in data:
		assert key in funcs, "Not found %s in funcs %s" % (key, list(funcs))
		item = funcs[key](data[key][0], baseDir)
		file.create_dataset(key, (N, *item.shape), dtype=item.dtype)
		file[key][0] = item

	# Do the rest N-1 items identically
	for i in range(1, N):
		print("%d/%d" % (i + 1, N), end="\r")
		for key in data:
			file[key][i] = funcs[key](data[key][i], baseDir)
	print("")

def getDataStatistics(args, file, maxDepthMeters=300):
	if args.test_export:
		statisticsFile = h5py.File(args.statistics_file, "r")
		statistics = h5ReadDict(statisticsFile["others"]["dataStatistics"])
	else:
		statistics = {}
		dataKeys = list(file)
		positions = {k : file[k]["position"][0 : ] for k in dataKeys}
		posConcat = np.concatenate([positions[k] for k in positions])
		statistics["position"] = {"min" : posConcat.min(axis=0), "max" : posConcat.max(axis=0)}
		statistics["depth"] = {"min" : 0, "max" : maxDepthMeters}
	return statistics

def main():
	args = getArgs()
	paths = getPaths(args.baseDir)
	print("Got %d paths. Keys: %s" % (len(paths["rgb"]), list(paths.keys())))

	paths = getTrainValPaths(paths, args.splits, args.split_keys, keepN=args.N)
	plotPaths(paths)

	file = h5py.File(args.resultFile, "w")
	for k in paths:
		file.create_group(k)
		print("Storing %s set" % (k))
		storeToH5File(args.baseDir, file[k], paths[k])

	print("Storing statistics!")
	if args.test_export:
		statisticsFile = h5py.File(args.statistics_file, "r")
		# This is here so the dataset reader works as intended.
		file["train"] = file["test"]
		file["validation"] = file["test"]
	else:
		statisticsFile = file
	statistics = getDataStatistics(args, statisticsFile)
	others = {"datatStatistics" : statistics}
	h5StoreDict(file, {"others" : others})

	file.flush()
	print("Done! Exported to %s." % (args.resultFile))

if __name__ == "__main__":
	main()
