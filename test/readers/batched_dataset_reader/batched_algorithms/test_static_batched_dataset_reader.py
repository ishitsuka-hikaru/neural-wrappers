import numpy as np
from overrides import overrides
from typing import Tuple, List, Any
from neural_wrappers.readers import StaticBatchedDatasetReader, DatasetItem, DatasetIndex
from neural_wrappers.utilities import getGenerators

import sys
sys.path.append("..")
from test_batched_dataset_reader import Reader as BaseReader

class TestStaticBatchedDatasetReader:
	def test_constructor_1(self):
		reader = StaticBatchedDatasetReader(BaseReader(), batchSize=1)
		assert not reader is None

	def test_getItem_1(self):
		reader = StaticBatchedDatasetReader(BaseReader(), batchSize=1)
		batches = reader.getBatches()
		item = reader[getBatchIndex(batches, 0)]
		rgb = item["data"]["rgb"]
		assert rgb.shape[0] == 1
		assert batches[0] == 1
		assert np.abs(rgb - reader.baseReader.dataset[0:1]).sum() < 1e-5

	def test_getItem_2(self):
		reader = StaticBatchedDatasetReader(BaseReader(), batchSize=1)
		batches = reader.getBatches()
		n = len(batches)
		for j in range(100):
			batchIndex = getBatchIndex(batches, j % n)
			batchItem = reader[batchIndex]
			rgb = batchItem["data"]["rgb"]
			index = getBatchIndex(batches, j % n)
			assert len(rgb) == batches[j % n]
			assert np.abs(rgb - reader.baseReader.dataset[index.start : index.stop]).sum() < 1e-5

	def test_iterateForever_1(self):
		reader = StaticBatchedDatasetReader(BaseReader(), batchSize=1)
		batchSizes = reader.getBatches()
		n = len(batchSizes)
		g = reader.iterateForever()
		for j, (batchItem, B) in enumerate(g):
			try:
				assert B == batchSizes[j % n]
			except Exception:
				breakpoint()
			rgb = batchItem["data"]["rgb"]
			index = getBatchIndex(batchSizes, j % n)
			assert np.abs(rgb - reader.baseReader.dataset[index.start : index.stop]).sum() < 1e-5

			if j == 100:
				break

	def test_getNumIterations_1(self):
		reader = StaticBatchedDatasetReader(BaseReader(), batchSize=1)
		N = getGenerators(reader, batchSize=1)[1]
		halfN = getGenerators(reader, batchSize=2)[1]
		assert N // 2 == halfN

	def test_iterateOneEpoch(self):
		baseReader = BaseReader()
		reader1 = StaticBatchedDatasetReader(baseReader, batchSize=1)
		reader2 = StaticBatchedDatasetReader(baseReader, batchSize=1)
		g1, n1 = getGenerators(reader1, batchSize=1, maxPrefetch=0)
		g2, n2 = getGenerators(reader2, batchSize=2, maxPrefetch=0)
		for i in range(n1):
			i1_0, b1_0 = next(g1)
			i1_1, b1_1 = next(g1)
			i2, b2 = next(g2)
			rgb1_0 = i1_0["data"]["rgb"]
			rgb1_1 = i1_1["data"]["rgb"]
			rgb2 = i2["data"]["rgb"]

			assert b1_0 == 1
			assert b1_1 == 1
			assert b2 == 2
			assert np.abs(rgb2[0] - rgb1_0[0]).sum() < 1e-5
			assert np.abs(rgb2[1] - rgb1_1[0]).sum() < 1e-5

def main():
	# TestStaticBatchedDatasetReader().test_constructor_1()
	TestStaticBatchedDatasetReader().test_iterateForever_1()

if __name__ == "__main__":
	main()