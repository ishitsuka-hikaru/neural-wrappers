import numpy as np
from typing import List
from ..dataset_types import *

# def getBatchIndex(batches:List[int], i:int) -> DatasetIndex:
# 	# batches = [1, 5, 4, 2] => cumsum = [0, 1, 6, 10, 12]
# 	cumsum = np.insert(np.cumsum(batches), 0, 0)
# 	# i = 2 => B = [6, 7, 8, 9]
# 	# batchIndex = np.arange(cumsum[i], cumsum[i + 1])
# 	try:
# 		batchIndex = slice(cumsum[i], cumsum[i + 1])
# 	except Exception as e:
# 		print(str(e))
# 		breakpoint()
# 	return batchIndex