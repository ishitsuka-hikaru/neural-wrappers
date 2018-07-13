import numpy as np
from neural_wrappers.readers import DatasetReader

class BasicReader(DatasetReader):
	def __init__(self, dataDims, labelDims, dataDimTransform={}, labelDimTransform={}, \
		dataNormalizer={}, labelNormalizer={}, dataAugTransform={}, labelAugTransform={}):
		super().__init__(None, dataDims, labelDims, dataDimTransform, labelDimTransform, dataNormalizer, \
			labelNormalizer, dataAugTransform, labelAugTransform)

def double(x):
	return x * 2

class TestDatasetReader:
	def test_construct_1(self):
		reader = BasicReader("rgb", "classes")
		assert len(reader.dataDimTransform) == len(reader.dataDims)
		assert len(reader.labelDimTransform) == len(reader.labelDims)
		assert list(reader.dataDimTransform.keys())[0] == "rgb"
		assert list(reader.labelDimTransform.keys())[0] == "classes"

		reader = BasicReader(["rgb"], "classes")
		assert len(reader.dataDimTransform) == len(reader.dataDims)
		assert len(reader.labelDimTransform) == len(reader.labelDims)
		assert list(reader.dataDimTransform.keys())[0] == "rgb"
		assert list(reader.labelDimTransform.keys())[0] == "classes"

		reader = BasicReader("rgb", ["classes"])
		assert len(reader.dataDimTransform) == len(reader.dataDims)
		assert len(reader.labelDimTransform) == len(reader.labelDims)
		assert list(reader.dataDimTransform.keys())[0] == "rgb"
		assert list(reader.labelDimTransform.keys())[0] == "classes"

		reader = BasicReader(["rgb"], ["classes"])
		assert len(reader.dataDimTransform) == len(reader.dataDims)
		assert len(reader.labelDimTransform) == len(reader.labelDims)
		assert list(reader.dataDimTransform.keys())[0] == "rgb"
		assert list(reader.labelDimTransform.keys())[0] == "classes"

		reader = BasicReader(dataDims=["rgb"], labelDims=["classes"], dataDimTransform={"rgb" : double})
		assert reader.dataDimTransform["rgb"] == double

		reader = BasicReader(dataDims=["rgb"], labelDims=["classes"], dataDimTransform=double)
		assert reader.dataDimTransform["rgb"] == double

		reader = BasicReader(dataDims=["rgb"], labelDims=["classes"], dataDimTransform=[double])
		assert reader.dataDimTransform["rgb"] == double

	def test_construct_2(self):
		reader = BasicReader(dataDims=["rgb"], labelDims=["classes"], dataDimTransform={"rgb" : double}, \
			dataNormalizer="min_max_normalization")
		assert len(reader.dataNormalizer) == 1
		assert reader.dataNormalizer["rgb"][0] == "min_max_normalization"
		assert len(reader.labelNormalizer) == 1
		assert reader.labelNormalizer["classes"][0] == "none"

		reader = BasicReader(dataDims=["rgb", "depth"], labelDims=["classes"], dataDimTransform={"rgb" : double}, \
			dataNormalizer={"rgb": "min_max_normalization"})
		assert len(reader.dataNormalizer) == 2
		assert reader.dataNormalizer["rgb"][0] == "min_max_normalization"
		assert reader.dataNormalizer["depth"][0] == "none"
