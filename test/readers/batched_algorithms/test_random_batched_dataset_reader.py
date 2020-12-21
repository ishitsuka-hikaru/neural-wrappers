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

	def test_getItem_1(self):
		reader = RandomBatchedDatasetReader(BaseReader())
		item, B = reader.getItem(0)
		rgb = item["data"]["rgb"]
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

	# def test_iterateForever_1(self):
	# 	reader = RandomBatchedDatasetReader(BaseReader())
	# 	generator = reader.iterateForever()
	# 	breakpoint()
	# 	n = 0
	# 	k = 0
	# 	batches = None
	# 	for j, (batchItem, B) in enumerate(reader.iterateForever()):
	# 		if k == 0 or k % n == 0:
	# 			oldBatches = batches
	# 			batches = reader.getBatches()
	# 			print("[tesT_iterate_forever_1] old: %s. new: %s" % (oldBatches, batches))
	# 			n = len(batches)
	# 			k = 0

	# 		rgb = batchItem["data"]["rgb"]
	# 		try:
	# 			assert B == batches[k % n]
	# 		except Exception:
	# 			print("j=", j, batches, n, "=>", B, batches[k % n])
	# 			# breakpoint()

	# 		index = reader.getBatchIndex(batches, k % n)
	# 		assert np.abs(rgb - reader.baseReader.dataset[index.start : index.stop]).sum() < 1e-5

	# 		k += 1
	# 		if j == 100:
	# 			break

	# def test_iterateOneEpoch_1(self):
	# 	reader = RandomBatchedDatasetReader(BaseReader())
	# 	assert reader.numShuffles == 1
	# 	g = reader.iterateOneEpoch()
	# 	N = reader.getNumIterations()
	# 	assert reader.numShuffles == 1
	# 	_ = next(g)
	# 	assert reader.numShuffles == 1 + (N == 1)

	# def test_getGenerators_2(self):
	# 	reader = RandomBatchedDatasetReader(BaseReader())
	# 	assert reader.numShuffles == 1
	# 	g, N = getGenerators(reader)
	# 	print("[test_getGenerators_2] N=%d" % N)
	# 	_ = next(g)
	# 	assert reader.numShuffles == 2

	# 	# for i in range(N-1):
	# 	# 	_ = next(g)
	# 	# assert reader.numShuffles == 3
	# 	# _ = next(g)
	# 	# assert reader.numShuffles == 

	# # TODO: Make this test pass :) We need to edit StaticBatchedDatasetReader to not make side effect change to the
	# #  underlying dataset reader, but instead use it accordingly
	# def test_iterateOneEpoch(self):
	# 	baseReader = BaseReader()
	# 	reader1 = StaticBatchedDatasetReader(baseReader, batchSize=1)
	# 	reader2 = StaticBatchedDatasetReader(baseReader, batchSize=1)
	# 	g1, n1 = getGenerators(reader1, batchSize=1, maxPrefetch=0)
	# 	g2, n2 = getGenerators(reader2, batchSize=2, maxPrefetch=0)
	# 	for i in range(n1):
	# 		i1_0, b1_0 = next(g1)
	# 		i1_1, b1_1 = next(g1)
	# 		i2, b2 = next(g2)
	# 		rgb1_0 = i1_0["data"]["rgb"]
	# 		rgb1_1 = i1_1["data"]["rgb"]
	# 		rgb2 = i2["data"]["rgb"]

	# 		assert b1_0 == 1
	# 		assert b1_1 == 1
	# 		assert b2 == 2
	# 		assert np.abs(rgb2[0] - rgb1_0[0]).sum() < 1e-5
	# 		assert np.abs(rgb2[1] - rgb1_1[0]).sum() < 1e-5

def main():
	TestRandomBatchedDatasetReader().test_iterateForever_1()

if __name__ == "__main__":
	main()