# Converter for CARLA simulator. Exports to H5 file based on a given path to an Unreal/Carla export (of pngs).
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
from PIL import Image
from neural_wrappers.utilities import h5StoreDict, h5ReadDict
from argparse import ArgumentParser
from functools import partial

def getArgs():
	parser = ArgumentParser()
	parser.add_argument("baseDir")
	parser.add_argument("resultFile")
	parser.add_argument("storeMethod")
	parser.add_argument("--N", type=int, default=None)
	parser.add_argument("--export_type", default="regular")
	parser.add_argument("--splits", default="80,20")
	parser.add_argument("--split_keys", default="train,validation")
	parser.add_argument("--statistics_file")

	args = parser.parse_args()
	assert args.storeMethod in ("h5", "paths")
	assert args.export_type in ("regular_old", "regular", "test")
	if args.export_type in ("regular_old", "regular"):
		args.split_keys = args.split_keys.split(",")
		args.splits = list(map(lambda x : float(x) / 100, args.splits.split(",")))
	elif args.export_type == "test":
		args.split_keys = ["test"]
		args.splits = [1]
		assert not args.statistics_file is None, \
			"For test export, we need the path to the train set, so we can copy its statistics"
	print("Export type: %s. Split keys: %s. Splits: %s. Stats file: %s" \
		% (args.export_type, args.split_keys, args.splits, args.statistics_file))

	assert abs(sum(args.splits) - 1) < 1e-5
	assert len(args.splits) == len(args.split_keys)
	return args

def checkPaths(baseDir, result):
	for key in result:
		if not result[key].dtype.char in {"S", "U"}:
			continue
		paths = list(map(lambda x : baseDir + str(x, "utf8"), result[key].flatten()))
		for path in paths:
			assert os.path.isfile(path), path

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

	# def rgbDomain2Func(rgbItem):
	# 	return rgbItem.replace("rgb", "rgbDomain2")

	def semanticFunc(rgbItem):
		return rgbItem.replace("rgb", "semanticSegmentation")

	def normalFunc(rgbItem):
		return rgbItem.replace("rgb", "normal")

	def cameraNormalFunc(rgbItem):
		return rgbItem.replace("rgb", "cameranormal")

	def wireframeFunc(rgbItem):
		return rgbItem.replace("rgb", "wireframe")

	def halftoneFunc(rgbItem):
		return rgbItem.replace("rgb", "halftone")

	def flowFunc(rgbItem, skip):
		# X and Y vectors are stored in 2 different files
		a = [rgbItem.replace("rgb", "flowr2"), rgbItem.replace("rgb", "flowr3")]
		return a

	rgbList = sorted(list(filter(lambda x : x.find("rgb_") != -1, os.listdir(baseDir))))
	N = len(rgbList)
	result = {
		"rgb" : np.array(rgbList, "S"),
		# "rgbDomain2" : np.array(list(map(rgbDomain2Func, rgbList)), "S"),
		"depth" : np.array(list(map(depthFunc, rgbList)), "S"),
		"position" : np.array(list(map(positionFunc, rgbList)), np.float32),
		"ids" : np.array(list(map(idsFunc, rgbList)), np.uint64),
		"semantic_segmentation" : np.array(list(map(semanticFunc, rgbList)), "S"),
		"normal" : np.array(list(map(normalFunc, rgbList)), "S"),
		"cameranormal" : np.array(list(map(cameraNormalFunc, rgbList)), "S"),
		"wireframe" : np.array(list(map(wireframeFunc, rgbList)), "S"),
		"halftone" : np.array(list(map(halftoneFunc, rgbList)), "S"),
		"optical_flow(t-1, t)" : np.array(list(map(partial(flowFunc, skip=1), rgbList)), "S"),
		# "optical_flow(t+2)" : np.array(list(map(partial(flowFunc, skip=2), rgbList)), "S"),
		# "optical_flow(t+3)" : np.array(list(map(partial(flowFunc, skip=3), rgbList)), "S"),
		# "optical_flow(t+4)" : np.array(list(map(partial(flowFunc, skip=4), rgbList)), "S")
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
	checkPaths(baseDir, result)
	return result

def getDataIndexes(splits, splitKeys, N):
	dataIx, lastIx = {}, 0
	for i in range(len(splitKeys) - 1):
		nK = int(splits[i] * N)
		dataIx[splitKeys[i]] = (lastIx, lastIx + nK)
		lastIx += nK
	dataIx[splitKeys[-1]] = (lastIx, N)
	return dataIx

# Return a dict of type
# 	{
# 		"train" : {"rgb" : [a, b), "depth" : [a, b), ...},
#		"test" : {"rgb" : [b, c), "depth" : [b, c), ...},
#		...
# 	}
def getSplitPaths(paths, splitKeys, pathKeys, dataIx, keepN):
	# Now, take the paths for all the keys as defined by indexes above
	newPaths = {}
	for k in splitKeys:
		startIx, endIx = dataIx[k]
		thisPaths = {pathKey : paths[pathKey][startIx : endIx] for pathKey in pathKeys}
		newPaths[k] = thisPaths

	if not keepN is None:
		for k in splitKeys:
			newPaths[k] = {pathKey : newPaths[k][pathKey][0 : keepN] for pathKey in pathKeys}

	for splitKey in splitKeys:
		thisData = newPaths[splitKey]
		firstValues = list(thisData.values())[0]
		N = len(firstValues)

		print("Key: %s. Range: %s. N=%d" % (splitKey, dataIx[splitKey], N))

	return newPaths

def getTrainValPaths(paths, splits, splitKeys, exportType, keepN=None):
	np.random.seed(42)
	N = len(paths["rgb"])
	# rgb, depth, semantic etc.
	pathKeys = paths.keys()
	# Get (startIndex, endIndex) tuple for each key
	dataIx = getDataIndexes(splits, splitKeys, N)
	print(dataIx)

	# Randomize order BEFORE splitting
	if exportType == "regular_old":
		perm = np.random.permutation(N)
		paths = {k : paths[k][perm] for k in pathKeys}

	splitPaths = getSplitPaths(paths, splitKeys, pathKeys, dataIx, keepN)
	# Randomize order AFTER splitting
	if exportType == "regular":
		for splitKey in splitKeys:
			N = dataIx[splitKey][1] - dataIx[splitKey][0]
			perm = np.random.permutation(N)
			for pathKey in pathKeys:
				splitPaths[splitKey][pathKey] = splitPaths[splitKey][pathKey][perm]

	return splitPaths

def plotPaths(paths):
	for k in paths:
		plt.gcf().clear()
		plt.scatter(paths[k]["position"][..., 0], paths[k]["position"][..., 1])
		print("Storing PNG paths: %s_points.png" % (k))
		plt.savefig("%s_points.png" % (k))

def storeToH5File(baseDir, file, data):
	# from neural_wrappers.readers.carla_h5_reader import CarlaH5PathsReader #TODO
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

def doStatistics(args, file):
	def computeStatistics(file):
		statistics = {}
		dataKeys = list(file)
		positions = {k : file[k]["position"][0 : ] for k in dataKeys}
		posConcat = np.concatenate([positions[k] for k in positions])
		statistics["position"] = {"min" : posConcat.min(axis=0), "max" : posConcat.max(axis=0)}
		return statistics

	print("[doStatistics] Storing statistics!")
	if args.statistics_file:
		print("[doStatistics] Using statistics of", args.statistics_file)
		statisticsFile = h5py.File(args.statistics_file, "r")
		statistics = h5ReadDict(statisticsFile["others"]["dataStatistics"])
	else:
		print("[doStatistics] Computing statistics (positions extremes, depth max) using this dataset.")
		statistics = computeStatistics(file)

	others = {"dataStatistics" : statistics, "baseDirectory" : os.path.abspath(args.baseDir)}
	h5StoreDict(file, {"others" : others})
	return file

def doWorkH5(baseDir, file, paths):
	for k in paths:
		file.create_group(k)
		print("Storing %s set" % (k))
		storeToH5File(baseDir, file[k], paths[k])

def doWorkPaths(file, paths):
	h5StoreDict(file, paths)

def doWork(args, file, paths):
	print("[doWork] Storing method:", args.storeMethod)
	if args.storeMethod == "h5":
		doWorkH5(args.baseDir, file, paths)
	elif args.storeMethod == "paths":
		doWorkPaths(file, paths)

	# This is here so the dataset reader works as intended.
	if args.export_type == "test":
		file["train"] = file["test"]
		file["validation"] = file["test"]
	return file

def main():
	args = getArgs()
	paths = getPaths(args.baseDir)
	print("Got %d paths. Keys: %s" % (len(paths["rgb"]), list(paths.keys())))

	paths = getTrainValPaths(paths, args.splits, args.split_keys, exportType=args.export_type, keepN=args.N)
	plotPaths(paths)

	file = h5py.File(args.resultFile, "w")
	file = doWork(args, file, paths)
	file = doStatistics(args, file)
	file.flush()
	print("Done! Exported to %s." % (args.resultFile))

if __name__ == "__main__":
	main()
