import numpy as np
from overrides import overrides
from typing import Any, Iterator, Dict, Callable
from neural_wrappers.readers import BatchedDatasetReader, DatasetReader

def rgbGetter(dataset, index):
	return dataset[index]

class Reader(DatasetReader):
	def __init__(self):
		super().__init__(
			dataBuckets = {"data" : ["rgb"]},
			dimGetter = {"rgb" : rgbGetter},
			dimTransform = {}
		)
		self.dataset = np.random.randn(10, 3)

	@overrides
	def getDataset(self) -> Any:
		return self.dataset

	@overrides
	def getNumData(self) -> int:
		return len(self.dataset)

	@overrides
	def getIndex(self, i):
		return 0

	# @overrides
	# def getItem(self, i):
	# 	return self.dataset[i]

# dataBuckets = {
# 	"data" : ["rgb", "depth"],
# 	"labels" : ["depth", "semantic"]
# }
# dimGetter = {
# 	"rgb" : lambda dataset, index: None,
# 	"depth" : lambda dataset, index: None,
# 	"semantic" : lambda dataset, index: None
# }
# dimTransform = {
# 	"data" : {
# 		"rgb" : lambda x : x,
# 		"depth" : lambda x : x
# 	},
# 	"labels" : {
# 		"depth" : lambda x : x,
# 	}
# }

class TestDatasetReader:
	def test_constructor_1(self):
		reader = Reader()
		item = reader.getItem(0)
		pass


def main():
	TestDatasetReader().test_constructor_1()

if __name__ == "__main__":
	main()