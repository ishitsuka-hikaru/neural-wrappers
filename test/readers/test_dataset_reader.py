import numpy as np
from neural_wrappers.readers import DatasetReader

class BasicReader(DatasetReader):
	def __init__(self, dataDims, labelDims, dimTransform={}, normalizer={}, augTransform=[], resizer={}):
		allDims = dataDims[:]
		allDims.extend(labelDims)
		super().__init__(None, allDims=allDims, dataDims=dataDims, labelDims=labelDims, dimTransform=dimTransform, \
			normalizer=normalizer, augTransform=augTransform, resizer=resizer)

def double(x):
	return x * 2

class TestDatasetReader:
	def test_construct_1(self):
		reader = BasicReader(["rgb"], ["classes"])
		keys = list(reader.dimTransform.keys())
		assert "rgb" in keys
		assert "classes" in keys

		reader = BasicReader(dataDims=["rgb"], labelDims=["classes"], dimTransform={"rgb" : double})
		assert reader.dimTransform["rgb"] == double

	def test_construct_2(self):
		reader = BasicReader(dataDims=["rgb"], labelDims=["classes"], dimTransform={"rgb" : double}, \
			normalizer={"rgb" : "min_max_normalization"})
		assert len(reader.normalizer) == 2
		assert reader.normalizer["rgb"][0] == "min_max_normalization"

		reader = BasicReader(dataDims=["rgb", "depth"], labelDims=["classes"], dimTransform={"rgb" : double}, \
			normalizer={"rgb": "min_max_normalization"})
		assert len(reader.normalizer) == 3
		assert reader.normalizer["rgb"][0] == "min_max_normalization"
		assert reader.normalizer["depth"][0] == "identity"
