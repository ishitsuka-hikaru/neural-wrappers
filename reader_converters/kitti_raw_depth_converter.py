# This converter only uses the RAW and the depth archives, and creates a h5py file using only the image_2 files.
# For object detection (no raw images), use the other converter.
# The structure of the h5py File is:
# - train
#     raw
#       rgb
#       depth
#     (TBD NAME: standard) -- this is not created by this file
#       rgb
#       labels (for object detection)
# - val
#     raw
#       rgb
#       depth
import h5py
import sys, os
from PIL import Image
import numpy as np
from lycon import resize, Interpolation

def flushPrint(message):
	sys.stdout.write(message + "\n")
	sys.stdout.flush()

def getPaths(baseRaw, baseDepth):
	paths = {"rgb" : [], "depth" : []}
	allDepthDirs = os.listdir(baseDepth)
	for Dir in allDepthDirs:
		depthDirPath = "%s/%s/proj_depth/groundtruth/image_02" % (baseDepth, Dir)
		allDepthFiles = os.listdir(depthDirPath)
		allDepthAbsPath = map(lambda x : depthDirPath + os.sep + x, allDepthFiles)
		paths["depth"].extend(allDepthAbsPath)

		# Find equivalent RGB for the depth path
		pngDirPath = "%s/%s/%s/image_02/data" % (baseRaw, Dir[0 : 10], Dir)
		allPngAbsPath = map(lambda x : pngDirPath + os.sep + x, allDepthFiles)
		paths["rgb"].extend(allPngAbsPath)
	return paths

def doDataset(file, baseRaw, baseDepth, mode):
	flushPrint("Doing %s" % (mode))
	paths = getPaths(baseRaw, baseDepth)
	numData = len(paths["rgb"])
	file.create_group(mode)
	file[mode].create_group("raw")
	group = file[mode]["raw"]

	dataShape = (numData, 375, 1224, 3)
	labelShape = (numData, 375, 1224)
	group.create_dataset("rgb", shape=dataShape, dtype=np.uint8)
	group.create_dataset("depth", shape=labelShape, dtype=np.uint16)

	shapes = {}
	for i in range(numData):
		if i % 500 == 0:
			flushPrint("%i/%i done." % (i, numData))

		image = np.uint8(Image.open(paths["rgb"][i]))
		image = resize(image, height=375, width=1224, interpolation=Interpolation.LANCZOS)
		group["rgb"][i] = image

		depth = np.uint16(Image.open(paths["depth"][i]))
		depth = resize(depth, height=375, width=1224, interpolation=Interpolation.LANCZOS)
		group["depth"][i] = depth

def main():
	assert len(sys.argv) == 3
	file = h5py.File("kitti.h5", "a")

	baseRaw = os.path.abspath(sys.argv[1])
	baseDepth = os.path.abspath(sys.argv[2])

	# Do the thingy
	doDataset(file, baseRaw, baseDepth + os.sep + "train" , "train")
	doDataset(file, baseRaw, baseDepth + os.sep + "val", "validation")


if __name__ == "__main__":
	main()