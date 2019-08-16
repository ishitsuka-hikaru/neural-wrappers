import numpy as np

RESIZE_LIB = "lycon"
assert RESIZE_LIB in ("lycon", "opencv", "pillow")

# @brief Generic function to resize ONE 2D image of shape (H, W) or 3D of shape (H, W, D) into (height, width [, D])
# @param[in] data Image (or any 2D/3D array)
# @param[in] height Desired resulting height
# @param[in] width Desired resulting width
# @param[in] interpolation method. Valid options are specific to the library used for the actual resizing.
# @return Resized image.
def resize(data, height, width, interpolation):
	global RESIZE_LIB
	assert RESIZE_LIB in ("lycon", ), "TODO: OpenCV and Pillow"

	funcs = {
		"lycon" : resize_lycon
	}

	return funcs[RESIZE_LIB](data, height, width, interpolation)

# @brief Lycon based image resizing function
# @param[in] height Desired resulting height
# @param[in] width Desired resulting width
# @param[in] interpolation method. Valid options: bilinear, nearest, cubic
# @return Resized image.
def resize_lycon(data, height, width, interpolation):
	from lycon import resize as resizeFn, Interpolation
	assert interpolation in ("bilinear", "nearest", "cubic")
	interpolationTypes = {
		"bilinear" : Interpolation.LINEAR,
		"nearest" : Interpolation.NEAREST,
		"cubic" : Interpolation.CUBIC
	}

	interpolationType = interpolationTypes[interpolation]
	return resizeFn(data, height=height, width=width, interpolation=interpolationType)

def resize_black_bars(data, height, width, interpolation):
	# No need to do anything if shapes are identical.
	assert len(data.shape) in (2, 3)
	if data.shape[0] == height and data.shape[1] == width:
		return np.copy(data)

	desiredShape = (height, width) if len(data.shape) == 2 else (height, width, data.shape[-1])

	imgH, imgW = data.shape[0 : 2]
	desiredH, desiredW = desiredShape[0 : 2]

	# Find the rapports between the imgH/desiredH and imgW/desiredW
	rH, rW = imgH / desiredH, imgW / desiredW

	# Find which one is the highest, that one will be used
	maxRapp = max(rH, rW)
	assert maxRapp != 0, "Cannot convert data of shape %s to (%d, %d)" % (data.shape, height, width)

	# Compute the new dimensions, based on the highest rapport
	scaledH, scaledW = int(imgH // maxRapp), int(imgW // maxRapp)
	assert scaledH != 0 and scaledW != 0, "Cannot convert data of shape %s to (%d, %d)" % (data.shape, height, width)
	resizedData = resize(data, height=scaledH, width=scaledW, interpolation=interpolation)

	# Also, find the half, so we can insert the other dimension from the half
	# Insert the resized image in the original image, halving the larger dimension and keeping half black bars in
	#  each side
	newData = np.zeros(desiredShape, dtype=data.dtype)
	halfH, halfW = int((desiredH - scaledH) // 2), int((desiredW - scaledW) // 2)
	newData[halfH : halfH + scaledH, halfW : halfW + scaledW] = resizedData
	return newData

# @brief Resizes a batch of data of shape: BxHxW(xD) to a desired shape of BxdWxdH(xD)
# @param[in] height Desired resulting height
# @param[in] width Desired resulting width
# @param[in] interpolation Interpolation method. Valid options: bilinear, nearest, cubic
# @param[in] mode What type of resizing to do. Valid values: default (can break proportions) or black_bars.
def resize_batch(data, height, width, interpolation="bilinear", mode="default"):
	assert len(data.shape) in (3, 4)
	assert mode in ("default", "black_bars")

	# No need to do anything if shapes are identical.
	if data.shape[1] == height and data.shape[2] == width:
		return np.copy(data)

	funcs = {
		"default" : resize,
		"black_bars" : resize_black_bars
	}

	numData = len(data)
	desiredShape = (height, width) if len(data.shape) == 2 else (height, width, data.shape[-1])
	newData = np.zeros((numData, *desiredShape), dtype=data.dtype)
	resizeFunc = funcs[mode]
	print("newData.shape", newData.shape)

	for i in range(len(data)):
		print("data[i].shape", data[i].shape)
		result = resizeFunc(data[i], height=height, width=width, interpolation=interpolation)
		print("result.shape", result.shape)
		print("desiredShape", desiredShape)
		# sys.exit(0)
		newData[i] = result.reshape(desiredShape)
	return newData