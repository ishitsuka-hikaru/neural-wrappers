# from .dataset_reader import DatasetReader, ClassificationDatasetReader
from .dataset_reader import DatasetReader
from .batched_reader import BatchedDatasetReader, StaticBatchedDatasetReader

# Composite readers (built on top of existing readers)
from .percent_dataset_reader import PercentDatasetReader
from .cached_dataset_reader import CachedDatasetReader

# from .classification_dataset_reader import ClassificationDatasetReader
# from .dataset_reader_old import ClassificationDatasetReader

from .internal import *
from .datasets import *

# from .corrupter_reader import CorrupterReader
# from .combined_dataset_reader import CombinedDatasetReader

# from .nyudepth_reader import NYUDepthReader
# from .citysim_reader import CitySimReader
# from .mnist_reader import MNISTReader
# from .cityscapes_reader import CityScapesReader
# from .cifar10_reader import Cifar10Reader
# from .kitti_reader import KITTIReader
# from .kitti_obj_reader import KITTIObjReader
# from .indoor_cvpr09_reader import IndoorCVPR09Reader
# from .word2vec_reader import Word2VecReader
# from .carla_h5_reader import CarlaH5Reader, CarlaH5PathsReader
