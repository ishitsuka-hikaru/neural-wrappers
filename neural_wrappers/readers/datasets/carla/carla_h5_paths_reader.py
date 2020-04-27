import numpy as np
from typing import Dict, Callable, List
from functools import partial
from ...h5_dataset_reader import H5DatasetReader
from ....utilities import tryReadImage
from neural_wrappers.utilities import resize_batch

def rgbNorm(x):
	# x [MBx854x854x3] => [MBx256x256x3] :: [0 : 255]
	x = resize_batch(x, height=256, width=256)
	# x :: [0 : 255] => [0: 1]
	x = x.astype(np.float32) / 255
	return x

def rgbReader(dataset, index, readerObj):
	baseDirectory = readerObj.dataset["others"]["baseDirectory"][()]
	paths = dataset["rgb"][index.start : index.end]

	results = []
	for path in paths:
		path = "%s/%s" % (baseDirectory, str(path, "utf8"))
		rgb = tryReadImage(path).astype(np.uint8)
		results.append(rgb)
	return np.array(results)

class CarlaH5PathsReader(H5DatasetReader):
	def __init__(self, datasetPath : str):#), dataBuckets : Dict[str, List[str]], \
		#dimTransform : Dict[str, Dict[str, Callable]]):
		# dimGetter : Dict[str, DimGetterCallable], 

		dataBuckets = {
			"data" : ["rgb"]
		}

		dimGetter = {
			"rgb" : partial(rgbReader, readerObj=self)
		}

		dimTransform ={
			"data" : {
				"rgb" : rgbNorm
			}
		}
		super().__init__(datasetPath, dataBuckets, dimGetter, dimTransform)
