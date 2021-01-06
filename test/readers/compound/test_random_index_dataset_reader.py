import numpy as np
from overrides import overrides
from typing import Tuple, List, Any
from neural_wrappers.readers import CachedDatasetReader, RandomIndexDatasetReader, \
	MergeBatchedDatasetReader, StaticBatchedDatasetReader
from neural_wrappers.utilities import getGenerators, deepCheckEqual

import sys
sys.path.append("..")
from batched_dataset_reader.test_batched_dataset_reader import Reader as BaseBatchedReader
from test_dataset_reader import DummyDataset as BaseReader

def mergeFn(x):
	rgbs = np.stack([y["data"]["rgb"] for y in x], axis=0)
	classes = np.stack([y["labels"]["class"] for y in x], axis=0)
	return {"data" : {"rgb" : rgbs}, "labels" : {"class" : classes}}

class TestRandomIndexDatasetReader:
	def test_constructor_1(self):
		reader = RandomIndexDatasetReader(BaseReader(), seed=42)
		assert not reader is None

	def test_iterateForever_1(self):
		reader = RandomIndexDatasetReader(BaseReader(), seed=42)
		generator = reader.iterateForever()
		N = 10
		items = []
		for i in range(N):
			epochItems = []
			for j in range(len(generator)):
				epochItems.append(next(generator))
			items.append(epochItems)

		# len(reader)^10 == 10^10 chance of this test not passing :)
		firstItems = items[0]
		equal = []
		for i in range(1, N):
			ixItems = items[i]
			check = deepCheckEqual(firstItems, ixItems)
			equal.append(check)
		assert sum(equal) != N - 1

class TestRandomIndexBatchedDatasetReader:
	def test_constructor_1(self):
		reader = RandomIndexDatasetReader(BaseBatchedReader(), seed=42)
		assert not reader is None

	def test_iterateForever_1(self):
		reader = RandomIndexDatasetReader(BaseBatchedReader(), seed=42)
		generator = reader.iterateForever()
		N = 10
		items = []
		for i in range(N):
			epochItems = []
			for j in range(len(generator)):
				epochItems.append(next(generator))
			items.append(epochItems)
		# len(reader)^10 == 10^10 chance of this test not passing :)
		firstItems = items[0]
		equal = []
		for i in range(1, N):
			ixItems = items[i]
			try:
				check = deepCheckEqual(firstItems, ixItems)
			except Exception as e:
				print(e)
				breakpoint()
			equal.append(check)
		assert sum(equal) != N - 1

	def test_iterateForever_2(self):
		reader = BaseReader()
		reader2 = RandomIndexDatasetReader(reader, 42)
		reader3 = MergeBatchedDatasetReader(reader2, mergeFn=mergeFn)
		reader4 = StaticBatchedDatasetReader(reader3, batchSize=10)
		generator = reader4.iterateForever()
		N = 10
		items = []
		for i in range(N):
			epochItems = []
			for j in range(len(generator)):
				epochItems.append(next(generator))
			items.append(epochItems)
		# len(reader)^10 == 10^10 chance of this test not passing :)
		firstItems = items[0]
		equal = []
		for i in range(1, N):
			ixItems = items[i]
			try:
				check = deepCheckEqual(firstItems, ixItems)
			except Exception as e:
				print(e)
				breakpoint()
			equal.append(check)
		assert sum(equal) != N - 1

def main():
	TestRandomIndexBatchedDatasetReader().test_iterateForever_2()

if __name__ == "__main__":
	main()