# Compound algorithms are algorithms that take in a DatasetReader and update one of its main methods. These can be
#  updating __getitem__ as in the case of CachedDatasetReader, which reads data from a cache object instead of the
#  reader itself, if the item is cached. PercentDatasetReader updates getNumData and works with a partial dataset
#  instead of the whole dataset.
from .cached_dataset_reader import CachedDatasetReader
from .percent_dataset_reader import PercentDatasetReader
from .random_index_dataset_reader import RandomIndexDatasetReader
from .combined_dataset_reader import CombinedDatasetReader
from .debug_reader import DebugReader