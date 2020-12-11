import time
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shutil
import tempfile
from datetime import datetime, timedelta
from overrides import overrides
from media_processing_lib.video import MPLVideo, tryReadVideo
from tqdm import trange
from pathlib import Path
from neural_wrappers.utilities import getGenerators, deepCheckEqual, npGetInfo, RunningMean
from neural_wrappers.readers import DatasetReader, CachedDatasetReader
from pycache import NpyFS, DictMemory

class Reader(DatasetReader):
	def __init__(self, baseDir):
		self.baseDir = baseDir
		self.videoList = sorted([str(x) for x in Path(baseDir).glob("*.mp4")])
		print(self.videoList)
		super().__init__(
			dataBuckets = {"data" : {"video"}}, \
			dimGetter = {"video" : (lambda d, r : [tryReadVideo(d[i]) for i in r])}, \
			dimTransform = {}
		)
		
	@overrides
	def getDataset(self, topLevel:str):
		return self.videoList

	@overrides
	def getBatchDatasetIndex(self, i : int, topLevel : str, batchSize : int):
		startIndex = i * batchSize
		endIndex = min((i + 1) * batchSize, self.getNumData(topLevel))
		assert startIndex < endIndex, "startIndex < endIndex. Got values: %d %d" % (startIndex, endIndex)
		return range(startIndex, endIndex)
	
	@overrides
	def getNumData(self, topLevel:str) -> int:
		return len(self.videoList)

def createDataset(baseDir, nVideos):
	path = Path(baseDir)
	if path.exists():
		if len([x for x in path.glob("*.mp4")]) == nVideos:
			return
		else:
			shutil.rmtree(baseDir)

	path.mkdir(exist_ok=True, parents=True)
	times = np.random.randint(100, 1000, size=(nVideos, ))
	fpss = np.random.randint(5, 60, size=(nVideos, ))
	videos = [np.random.randint(0, 255, size=(time, 64, 64, 3)).astype(np.uint8) for time in times]
	videos = [MPLVideo(data=video, fps=float(fps)) for video, fps in zip(videos, fpss)]
	for i, video in enumerate(videos):
		video.save("%s/%d.mp4" % (baseDir, i))
	print("Dataset created at %s (%d vids)" % (baseDir, nVideos))

class TestCachedReader:
	def test_cached_reader_npy(self):
		nVideos = 5
		N = 2
		baseDir = "/tmp/test_cached_reader"
		cacheDir = "%s/.cache" % baseDir
		createDataset(baseDir, nVideos)
		reader = Reader(baseDir)
		readerNpyFS = CachedDatasetReader(reader, NpyFS(cacheDir))

		g1 = getGenerators(reader, batchSize=-1, keys=["train"])[0]
		g2 = getGenerators(readerNpyFS, batchSize=-1, keys=["train"])[0]

		# First time both should compute
		item1 = next(g1)
		item2 = next(g2)
		assert deepCheckEqual(item1, item2)

		# Second time only basic reader should compute
		item1 = next(g1)
		item2 = next(g2)
		assert deepCheckEqual(item1, item2)
		shutil.rmtree(cacheDir)

	def test_cached_dict_memory(self):
		nVideos = 5
		N = 2
		baseDir = "/tmp/test_cached_reader"
		createDataset(baseDir, nVideos)
		reader = Reader(baseDir)
		readerDictMemory = CachedDatasetReader(reader, DictMemory())

		g1 = getGenerators(reader, batchSize=-1, keys=["train"])[0]
		g2 = getGenerators(readerDictMemory, batchSize=-1, keys=["train"])[0]

		# First time both should compute
		item1 = next(g1)
		item2 = next(g2)
		assert deepCheckEqual(item1, item2)

		# Second time only basic reader should compute
		item1 = next(g1)
		item2 = next(g2)
		assert deepCheckEqual(item1, item2)

	def test_cached_reader_npy_build_cache(self):
		nVideos = 5
		N = 2
		baseDir = "/tmp/test_cached_reader"
		cacheDir = "%s/.cache" % baseDir
		createDataset(baseDir, nVideos)
		reader = Reader(baseDir)
		readerNpyFS = CachedDatasetReader(reader, NpyFS(cacheDir))

		g1 = getGenerators(reader, batchSize=-1, keys=["train"])[0]
		g2 = getGenerators(readerNpyFS, batchSize=-1, keys=["train"])[0]

		# First time both should compute
		item1 = next(g1)
		item2 = next(g2)
		assert deepCheckEqual(item1, item2)

		# Second time only basic reader should compute
		item1 = next(g1)
		item2 = next(g2)
		assert deepCheckEqual(item1, item2)
		shutil.rmtree(cacheDir)

	def test_cached_dict_memory_build_cache(self):
		nVideos = 5
		N = 2
		baseDir = "/tmp/test_cached_reader"
		createDataset(baseDir, nVideos)
		reader = Reader(baseDir)
		readerDictMemory = CachedDatasetReader(reader, DictMemory(), buildCache=True)

		g1 = getGenerators(reader, batchSize=-1, keys=["train"])[0]
		g2 = getGenerators(readerDictMemory, batchSize=-1, keys=["train"])[0]

		# First time both should compute
		item1 = next(g1)
		item2 = next(g2)
		assert deepCheckEqual(item1, item2)

		# Second time only basic reader should compute
		item1 = next(g1)
		item2 = next(g2)
		assert deepCheckEqual(item1, item2)
