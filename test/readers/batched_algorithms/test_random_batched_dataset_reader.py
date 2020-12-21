import numpy as np
from overrides import overrides
from typing import Tuple, List, Any
from neural_wrappers.readers import RandomBatchedDatasetReader, DatasetItem, DatasetIndex
from neural_wrappers.utilities import getGenerators

import sys
sys.path.append("..")
from test_batched_dataset_reader import Reader as BaseReader

class TestRandomBatchedDatasetReader:
	def test_constructor_1(self):
		reader = RandomBatchedDatasetReader(BaseReader())
		assert not reader is None

	def test_getBatchedItem_1(self):
		reader = RandomBatchedDatasetReader(BaseReader())
		batches = reader.getBatches()
		item = reader.getBatchItem(reader.getBatchIndex(batches, 0))
		rgb = item["data"]["rgb"]
		B = batches[0]
		assert rgb.shape[0] == B
		assert np.abs(rgb - reader.baseReader.dataset[0:B]).sum() < 1e-5

	def test_getBatchItem_1(self):
		reader = RandomBatchedDatasetReader(BaseReader())
		batches = reader.getBatches()
		n = len(batches)
		for j in range(100):
			index = reader.getBatchIndex(batches, j % n)
			batchItem = reader.getBatchItem(index)
			rgb = batchItem["data"]["rgb"]
			assert rgb.shape[0] == batches[j % n], "%d vs %d" % (rgb.shape[0], batches[j % n])
			assert np.abs(rgb - reader.baseReader.dataset[index.start : index.stop]).sum() < 1e-5

	def test_iterateForever_1(self):
		reader = RandomBatchedDatasetReader(BaseReader())
		generator = reader.iterateForever()
		k = 0
		for j, (batchItem, B) in enumerate(generator):
			if k == 0 or k % n == 0:
				n = len(generator)
				currentBatch = generator.currentGenerator.batches
				k = 0

			rgb = batchItem["data"]["rgb"]
			print("j=%d. batches: %s. rgb:%s" % (j, currentBatch, rgb.shape))
			index = reader.getBatchIndex(currentBatch, k % n)

			assert B == currentBatch[k % n]
			assert np.abs(rgb - reader.baseReader.dataset[index.start : index.stop]).sum() < 1e-5

			k += 1
			if j == 100:
				break

	def test_iterateOneEpoch_1(self):
		reader = RandomBatchedDatasetReader(BaseReader())
		assert reader.numShuffles == 0
		g = reader.iterateOneEpoch()
		N = len(g)
		print(reader.numShuffles)
		assert reader.numShuffles == 1
		_ = next(g)
		assert reader.numShuffles == 1

	def test_getGenerators_1(self):
		reader = RandomBatchedDatasetReader(BaseReader())
		assert reader.numShuffles == 0
		g, N = getGenerators(reader)
		_ = next(g)
		assert reader.numShuffles == 1

		for i in range(N-1):
			_ = next(g)
		assert reader.numShuffles == 1
		_ = next(g)
		assert reader.numShuffles == 2

	# Two generators going side by side 1 random epoch
	def test_getGenerators_2(self):
		reader = RandomBatchedDatasetReader(BaseReader(N=100))
		g1, n1 = getGenerators(reader)
		g2, n2 = getGenerators(reader)

		i1, i2 = 0, 0
		leftover1, leftover2 = np.zeros((0, 3)), np.zeros((0, 3))
		# while i1 < n1 or i2 < n2:
		for i in range(min(n1, n2)):
			item1, b1 = next(g1)
			item2, b2 = next(g2)
			rgb1 = item1["data"]["rgb"]
			rgb2 = item2["data"]["rgb"]

			rgb1 = np.concatenate([leftover1, rgb1])
			rgb2 = np.concatenate([leftover2, rgb2])
			Min = min(len(rgb1), len(rgb2))
			assert np.abs(rgb1[0 : Min] - rgb2[0 : Min]).sum() < 1e-5

			leftover1 = rgb1[Min :]
			leftover2 = rgb2[Min :]
			i1 += b1
			i2 += b2

		i += 1
		for j in range(i, n1):
			item1, b1 = next(g1)
			rgb1 = item1["data"]["rgb"]
			assert np.abs(rgb1 - leftover2[0 : b1]).sum() < 1e-5
			leftover2 = leftover2[b1 :]

		for j in range(i, n2):
			item2, b2 = next(g2)
			rgb2 = item2["data"]["rgb"]
			assert np.abs(rgb2 - leftover1[0 : b2]).sum() < 1e-5
			leftover1 = leftover1[b2 :]
		assert len(leftover1) == 0
		assert len(leftover2) == 0

def main():
	TestRandomBatchedDatasetReader().test_getGenerators_2()

if __name__ == "__main__":
	main()