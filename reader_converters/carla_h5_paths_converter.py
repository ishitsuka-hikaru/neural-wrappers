import h5py
import os
from carla_h5_converter import getArgs, getTrainValPaths, getPaths, plotPaths, getDataStatistics
from neural_wrappers.utilities import h5StoreDict

def main():
	args = getArgs()
	paths = getPaths(args.baseDir)
	print("Got %d paths. Keys: %s" % (len(paths["rgb"]), list(paths.keys())))

	paths = getTrainValPaths(paths, args.splits, args.split_keys, keepN=args.N)
	plotPaths(paths)

	file = h5py.File(args.resultFile, "w")
	h5StoreDict(file, paths)

	print("Storing statistics!")
	statisticsFile = h5py.File(args.statistics_file, "r") if args.test_export else file
	statistics = getDataStatistics(args, statisticsFile)
	baseDirectory = os.getcwd() + os.sep + args.baseDir
	others = {"dataStatistics" : statistics, "baseDirectory" : baseDirectory}
	h5StoreDict(file, {"others" : others})

	file.flush()
	print("Done! Exported to %s." % (args.resultFile))

if __name__ == "__main__":
    main()