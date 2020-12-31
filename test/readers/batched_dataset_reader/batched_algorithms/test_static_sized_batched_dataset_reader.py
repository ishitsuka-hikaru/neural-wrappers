import numpy as np
from overrides import overrides
from typing import Tuple, List, Any
from neural_wrappers.readers import StaticSizedBatchedDatasetReader, DatasetItem, DatasetIndex
from neural_wrappers.utilities import getGenerators

import sys
sys.path.append("..")
from test_batched_dataset_reader import Reader as BaseReader

class TestStaticSizedBatchedDatasetReader:
	def test_constructor_1(self):
		reader = StaticSizedBatchedDatasetReader(BaseReader(), batchSize=1)
		assert not reader is None

	def test_getItem_1(self):
		reader = StaticSizedBatchedDatasetReader(BaseReader(), batchSize=1)
		batches = reader.getBatches()
		item = reader[batches[0]]
		rgb = item["data"]["rgb"]
		assert rgb.shape[0] == 1
		assert reader.batchLens[0] == 1
		assert np.abs(rgb - reader.baseReader.dataset[0:1]).sum() < 1e-5

	def test_getItem_2(self):
		reader = StaticSizedBatchedDatasetReader(BaseReader(), batchSize=1)
		batches = reader.getBatches()
		n = len(batches)
		for j in range(100):
			batchIndex = batches[j % n]
			batchItem = reader[batchIndex]
			rgb = batchItem["data"]["rgb"]
			index = batches[j % n]
			assert len(rgb) == reader.batchLens[j % n]
			assert np.abs(rgb - reader.baseReader.dataset[index.start : index.stop]).sum() < 1e-5

	def test_iterateForever_1(self):
		reader = StaticSizedBatchedDatasetReader(BaseReader(), batchSize=1)
		g = reader.iterateForever()
		batches = g.batches
		n = len(batches)
		for j, (batchItem, B) in enumerate(g):
			try:
				assert B == g.batchLens[j % n]
			except Exception:
				pass
				# breakpoint()
			rgb = batchItem["data"]["rgb"]
			index = batches[j % n]
			assert np.abs(rgb - reader.baseReader.dataset[index.start : index.stop]).sum() < 1e-5

			if j == 100:
				break

	def test_getNumIterations_1(self):
		reader = BaseReader()
		reader1 = StaticSizedBatchedDatasetReader(reader, batchSize=1)
		reader2 = StaticSizedBatchedDatasetReader(reader, batchSize=2)
		g1, N1 = getGenerators(reader1)
		g2, N2 = getGenerators(reader2)

		assert N1 // 2 == N2

	def test_iterateOneEpoch(self):
		baseReader = BaseReader()
		reader1 = StaticSizedBatchedDatasetReader(baseReader, batchSize=1)
		reader2 = StaticSizedBatchedDatasetReader(baseReader, batchSize=2)
		g1, n1 = getGenerators(reader1, maxPrefetch=0)
		g2, n2 = getGenerators(reader2, maxPrefetch=0)
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
	# TestStaticSizedBatchedDatasetReader().test_constructor_1()
	TestStaticSizedBatchedDatasetReader().test_getNumIterations_1()

if __name__ == "__main__":
	main()