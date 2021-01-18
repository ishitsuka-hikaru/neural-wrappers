import sys
import os
import numpy as np
from neural_wrappers.readers import CombinedDatasetReader

sys.path.append(os.path.realpath(os.path.abspath(os.path.dirname(__file__))) + "/..")
from batched_dataset_reader.test_batched_dataset_reader import Reader as BaseBatchedReader
from test_dataset_reader import DummyDataset as BaseReader

class TestCombinedDatasetReader:
	def test_constructor_1(self):
		reader = CombinedDatasetReader([BaseReader(), BaseReader()])
		assert not reader is None

	def test_iterate_constructor_1(self):
		reader1 = BaseReader()
		reader2 = BaseReader(N=20)
		reader = CombinedDatasetReader([reader1, reader2])

		i1 = reader1.iterate()
		i2 = reader2.iterate()
		i = reader.iterate()
		assert not i1 is None
		assert not i2 is None
		assert not i is None

	def test_len_1(self):
		reader1 = BaseReader()
		reader2 = BaseReader(N=20)
		reader = CombinedDatasetReader([reader1, reader2])

		i1 = reader1.iterate()
		i2 = reader2.iterate()
		i = reader.iterate()
		assert len(i) == len(i1) + len(i2)

class TestCombinedBatchedDatasetReader:
	def test_constructor_1(self):
		reader = CombinedDatasetReader([BaseBatchedReader(), BaseBatchedReader()])
		assert not reader is None


if __name__ == "__main__":
	# TestCombinedBatchedDatasetReader().test_constructor_1()
	TestCombinedDatasetReader().test_iterate_constructor_1()
