import numpy as np
from typing import List

def batchIndexFromBatchSizes(batchSizes:List[int]) -> List[slice]:
    # batchSizes = [4, 1, 2, 3], so batch[0] has a size of 4, batch[2] a size of 2 etc.
    # actual batches are obtained by cumsum on lens: [0, 4, 5, 7, 10]. cumsum[0] = [0, 4), cumsum[1] = [4, 5) etc.
    cumsum = np.insert(np.cumsum(batchSizes), 0, 0)
    # We can further slice these batches for faster access
    batches = [slice(cumsum[i], cumsum[i + 1]) for i in range(len(cumsum) - 1)]
    return batches
