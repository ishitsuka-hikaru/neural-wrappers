import h5py
import numpy as np
import os
import pims
from .dataset_reader import DatasetReader
from neural_wrappers.transforms import Transformer
from neural_wrappers.utilities import resize_batch
from datetime import datetime

class CityScapesReader(DatasetReader):
	# @param[in] datasetPath The path to the root directory of the CityScapes videos
	# @param[in] imageShape The shape to which the images are resized during the iteration phase. Frames have shape
	#  870x1820x3, but normal values are 512x1024x3 (some networks require power of 2 inputs).
	# @param[in] labelShape The shape of the depths (TOOD: a list including shape of flow as well). Frames have a shape
	#  of 870x1820, but normal values are 256x512, which is the network output for 512x1024x3 input.
	# @param[in] skipFrames Each video is consistent of multiple 30-frames scenes (1 second at 30fps). For videos, we
	# want to process every frame (so skipFrames is 1 by default). For some other type of processing (like training a
	#  non-recurrent neural network), we might want to skip some frames (like get only every nth frame), so this value
	#  is higher. It cannot exceed 30, because there are only 30 frames in every scene and each scene is an independent
	#  unit of work.
	# @param[in] transforms A list of transformations to be applied on each iteration.
	# @param[in] dataSplit A list/tuple that sums to 100 representing the split between train/test/validation sets.
	#  The split is done based on the videos (so for 80/0/20, 80% of videos are used for training, 0% for testing and
	#  20% for validation). This does not take into account the size of the videos. So the 80% of the videos could be
	#  much smaller in number of frames than the last 20%. Depends on the dataset. For CityScapes, most of videos are
	#  600 frames long, with some final videos of each scenes being smaller (varying from 30 to 570).
	# @param[in] precomputedDurations If set true, the totalDurations and durations arrays are pre-computed. This is
	#  only valid for the current video configuration, but adding more videos or changing any video in any way that
	#  modifies the number of frames will result in the values to be invalid.
	# @param[in] flowAlgorithm An optical flow algorithm for each frame. Valid algorithms are "farneback" (requires
	#  OpenCV) and "deepflow2" (requires DeepFlow2 from NVIDIA with pre-computted weights). The required libraries can
	#  be omitted if the flows were pre-computted and stored as vidoes, which can be loaded on demand
	#  (see next parameter).
	# @param[in] useStoredFlow The flow algorithm can either computed on demand (requiring the library to be installed)
	#  or loaded from a pre-computted video.
	def __init__(self, datasetPath, imageShape, labelShape, skipFrames=1, transforms=["none"], dataSplit=(80, 0, 20), \
		flowAlgorithm=None, useStoredFlow=None):
		assert skipFrames > 0 and skipFrames <= 30
		assert flowAlgorithm is None or (not flowAlgorithm is None and type(useStoredFlow) is bool)
		self.datasetPath = datasetPath
		self.imageShape = imageShape
		self.labelShape = labelShape
		self.transforms = transforms
		self.dataSplit = dataSplit
		self.skipFrames = skipFrames
		self.flowAlgorithm = flowAlgorithm
		self.useStoredFlow = useStoredFlow

		self.dataAugmenter = Transformer(transforms, dataShape=imageShape, labelShape=labelShape)
		self.validationAugmenter = Transformer(["none"], dataShape=imageShape, labelShape=labelShape)
		self.setup()

	def setup(self):
		videosPath = self.datasetPath + os.sep + "video"
		depthPath = self.datasetPath + os.sep + "depth"
		flowFarnebackPath = self.datasetPath + os.sep + "flow_farneback"
		flowDeepFlow2Path = self.datasetPath + os.sep + "flow_deepflow2"

		allVideos = os.listdir(videosPath)
		allVideos = sorted(list(filter(lambda x : x.endswith("mp4"), allVideos)))
		numVideos = len(allVideos)
		assert numVideos > 0

		self.indexes, self.numData = self.computeIndexesSplit(numVideos)

		# Using the indexes, find all the paths to all the videos of all types for all labels. These will be used
		#  during iteration to actually read the videos and access the frame for training/testing.
		self.paths = {
			"train" : {
				"video" : [],
				"depth" : [],
				"flow_farneback" : [],
				"flow_deepflow2" : []
			},
			"test" : {
				"video" : [],
				"depth" : [],
				"flow_farneback" : [],
				"flow_deepflow2" : []
			},
			"validation" : {
				"video" : [],
				"depth" : [],
				"flow_farneback" : [],
				"flow_deepflow2" : []
			}
		}

		self.totalDurations = {
			"train" : 0,
			"test" : 0,
			"validation" : 0
		}

		self.durations = {
			"train" : None,
			"test" : None,
			"validation" : None
		}

		# Data structure that holds references to actual video objects, pre-loaded so they can be fast accessed.
		# It will be populated during iterate_once method, first time a video is requested (using self.paths).
		self.videos = {
			"train" : {
				"video" : [],
				"depth" : [],
				"flow_farneback" : [],
				"flow_deepflow2" : []
			},
			"test" : {
				"video" : [],
				"depth" : [],
				"flow_farneback" : [],
				"flow_deepflow2" : []
			},
			"validation" : {
				"video" : [],
				"depth" : [],
				"flow_farneback" : [],
				"flow_deepflow2" : []
			}
		}

		for Type in ["train", "test", "validation"]:
			typeVideos = allVideos[self.indexes[Type][0] : self.indexes[Type][1]]
			self.durations[Type] = np.zeros((len(typeVideos), ), dtype=np.uint32)
			for i in range(len(typeVideos)):
				videoName = typeVideos[i]
				depthName = depthPath + os.sep + "depth_" + videoName[6 : ]
				flowFarnebackName = flowFarnebackPath + os.sep + "flow_farneback_" + videoName[6 : -4] + ".avi"
				flowDeepFlow2Name = flowDeepFlow2Path + os.sep + "flow_deepflow2_" + videoName[6 : -4] + ".avi"
				videoName = videosPath + os.sep + videoName

				self.paths[Type]["video"].append(videoName)
				self.paths[Type]["depth"].append(depthName)
				if self.flowAlgorithm == "farneback":
					self.paths[Type]["flow_farneback"].append(flowFarnebackName)
				if self.flowAlgorithm == "deepflow2":
					self.paths[Type]["flow_deepflow2"].append(videoName)

				# Also save the duration, so we can compute the number of iterations (Takes about 10s for all dataset)
				video = pims.Video(videoName)
				# Some videos have 601 frames, instead of 600, so remove that last one (also during iteration)
				videoLen = len(video) - (len(video) % 30)
				self.totalDurations[Type] += videoLen
				self.durations[Type][i] = videoLen

		print(("[CityScapes Reader] Setup complete. Num videos: %d. Train: %d, Test: %d, Validation: %d. " + \
			"Frame shape: %s. Labels shape: %s. Frames: Train: %d, Test: %d, Validation: %d. Skip frames: %d") % \
			(numVideos, self.numData["train"], self.numData["test"], self.numData["validation"], self.imageShape, \
			self.labelShape, self.totalDurations["train"], self.totalDurations["test"], \
			self.totalDurations["validation"], self.skipFrames))

	def getNumIterations(self, type, miniBatchSize, accountTransforms=False):
		N = self.numData[type] // miniBatchSize + (self.numData[type] % miniBatchSize != 0)
		numIterations = 0
		thisDurations = self.durations[type]
		for i in range(N):
			startIndex = i * miniBatchSize
			endIndex = min((i + 1) * miniBatchSize, self.numData[type])
			thisVideosDurations = thisDurations[startIndex : endIndex]
			# [30, 60, 450, 420, 60] => 450 iterations for this batch, but the other elements will only contribute
			#  for their amount (so 30, 60, 420, 60). Thus, for first 30 iterations we get 5 items from this batch,
			#  then until 60 we get 4 items, then until 420 we get 2 items and for 420-450 we just get 1 item.
			maxDuration = np.max(thisVideosDurations)
			batchContribution = int(np.ceil(maxDuration / self.skipFrames))
			numIterations += batchContribution
		return numIterations if accountTransforms == False else numIterations * len(self.transforms)

	def getOpticalFlow(self, video, flowVideo, thisFrame, frameIndex):
		assert self.useStoredFlow == True
		# TODO: useStoredflow == False requires to compute the flow from this frame. If it's last frame, use previous
		#  frame for flow disparity, otherwise always use the next one.
		# if frameIndex % 30 == 29 => frameIndex -= 1
		frameIndex -= int(((frameIndex % 30) // 29))
		frames = flowVideo[frameIndex]
		return frames[..., 0 : 2]

	# Accesses videos[startIndex : endIndex]. If they don't exist, loads from paths[startIndex : endIndex], so the
	#  next time they're called, the videos are already loaded.
	def getOrLoadVideo(self, videos, paths, startIndex, endIndex):
		assert startIndex >= 0 and endIndex > startIndex
		res = videos[startIndex : endIndex]
		if len(res) != endIndex - startIndex:
			# print("Loading videos %d to %d" % (startIndex, endIndex))
			videos[startIndex : endIndex] = [pims.PyAVReaderIndexed(paths[j]) for j in range(startIndex, endIndex)]
		else:
			pass
			# print("Already loaded videos %d to %d" % (startIndex, endIndex))
		return videos[startIndex : endIndex]

	def iterate_once(self, type, miniBatchSize):
		assert type in ("train", "test", "validation")
		augmenter = self.dataAugmenter if type == "train" else self.validationAugmenter

		N = self.numData[type] // miniBatchSize + (self.numData[type] % miniBatchSize != 0)
		thisVideos = self.videos[type]
		thisPaths = self.paths[type]
		thisDurations = self.durations[type]

		for i in range(N):
			startIndex = i * miniBatchSize
			endIndex = min((i + 1) * miniBatchSize, self.numData[type])
			videos = self.getOrLoadVideo(thisVideos["video"], thisPaths["video"], startIndex, endIndex)
			depth_videos = self.getOrLoadVideo(thisVideos["depth"], thisPaths["depth"], startIndex, endIndex)
			if not self.flowAlgorithm is None and self.useStoredFlow == True:
				# For each flow algorithm that has a pre-computted flow video, load that video in memory
				# All the other flows that are not loaded here will be computed manually, and the library is expected
				#  to be installed for each of them.
				flowType = "flow_" + self.flowAlgorithm
				flow_videos = self.getOrLoadVideo(thisVideos[flowType], thisPaths[flowType], startIndex, endIndex)
			print("Here2?")

			numVideos = len(videos)
			frameShape, depthFrameShape = videos[0].frame_shape[0 : 2], depth_videos[0].frame_shape[0 : 2]
			# If no flow algorithm is used, just 3 channels (RGB) are needed. Otherwise, the flow is concatenated
			#  with the RGB image, so we get 5 channels: R, G, B, U, V
			numChannels = 3 if self.flowAlgorithm is None else 5
			images = np.zeros((numVideos, *frameShape, numChannels), dtype=np.float32)
			# Depths are 3 grayscale identical channels (due to necessity of saving them as RGB)
			depths = np.zeros((numVideos, *depthFrameShape), dtype=np.float32)

			thisVideosDurations = thisDurations[startIndex : endIndex]
			maxDuration = np.max(thisVideosDurations)

			for frame_i in range(0, maxDuration, self.skipFrames):
				validIndex = 0
				for video_i in range(numVideos):
					if thisVideosDurations[video_i] <= frame_i:
						continue

					image = videos[video_i][frame_i]
					images[validIndex][..., 0 : 3] = image / 255
					depths[validIndex] = depth_videos[video_i][frame_i][..., 0] / 255

					if not self.flowAlgorithm is None:
						flowVideo = flow_videos[video_i] if self.useStoredFlow == True else None
						flow = self.getOpticalFlow(videos[video_i], flowVideo, image, frame_i)
						images[validIndex][..., 3 : 5] = flow / 255
					validIndex += 1

				augGenerator = augmenter.applyTransforms(images[0 : validIndex], depths[0 : validIndex], "bilinear")
				for augmentedImages, augmentedDepths in augGenerator:
					yield augmentedImages, augmentedDepths
					del augmentedImages, augmentedDepths
			del videos
