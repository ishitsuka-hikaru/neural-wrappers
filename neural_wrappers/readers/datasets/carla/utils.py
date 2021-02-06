import numpy as np
import transforms3d.euler as txe
import h5py
from typing import Callable

from .carla_h5_paths_reader import CarlaH5PathsReader
from ....utilities import npCloseEnough, npGetInfo, smartIndexWrapper

# def opticalFlowReader(dataset:h5py._hl.group.Group, index:range, \
# 	dim:str, rawReadFunction:Callable[[str], np.ndarray]) -> np.ndarray:
# 	baseDirectory = dataset.file["others"]["baseDirectory"][()]
# 	paths = dataset[dim][index.start : index.stop]
# 	breakpoint()

# 	results = []
# 	for path in paths:
# 		path_x, path_y = path
# 		path_x, path_y = "%s/%s" % (baseDirectory, str(path_x, "utf8")), "%s/%s" % (baseDirectory, str(path_y, "utf8"))
# 		flow_x, flow_y = readerObj.rawFlowReadFunction(path_x), readerObj.rawFlowReadFunction(path_y)
# 		flow = np.stack([flow_x, flow_y], axis=-1)
# 		results.append(flow)
# 	return np.array(results)

# def rgbNeighbourReader(dataset:h5py._hl.group.Group, index:range, \
# 	skip:int, readerObj:CarlaH5PathsReader) -> np.ndarray:
# 	baseDirectory = dataset.file["others"]["baseDirectory"][()]

# 	# For optical flow we have the problem that the flow data for t->t+1 is stored at index t+1, which isn't
# 	#  necessarily 1 index to the right (trian set may be randomized beforehand). Thus, we need to get the indexes
# 	#  of the next neighbours of this top level (train/test etc.), and then read the paths at those indexes.
# 	topLevel = readerObj.getActiveTopLevel()
# 	key = "t+%d" % (skip)
# 	neighbourIds = readerObj.idOfNeighbour[topLevel][key][index.start : index.stop]
# 	paths = smartIndexWrapper(dataset["rgb"], neighbourIds)

# 	results = []
# 	for path in paths:
# 		path = "%s/%s" % (baseDirectory, str(path, "utf8"))
# 		results.append(readerObj.rawReadFunction(path))
# 	return np.array(results)

# def depthReadFunction(path:str, readerObj:CarlaH5PathsReader) -> np.ndarray:
# 	return readerObj.rawDepthReadFunction(path)
