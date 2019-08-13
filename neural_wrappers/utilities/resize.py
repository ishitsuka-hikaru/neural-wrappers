RESIZE_TYPE = "lycon"
assert RESIZE_TYPE in ("lycon", "opencv", "pillow")

def resize(data, height, width, interpolation):
	global RESIZE_TYPE

	if RESIZE_TYPE == "lycon":
		return resize_lycon(data, height, width, interpolation)
	elif RESIZE_TYPE == "opencv":
		assert False, "TODO"
	elif RESIZE_TYPE == "pillow":
		assert False, "TODO"

def resize_lycon(data, height, width, interpolation):
	from lycon import resize as resizeFn, Interpolation
	assert interpolation in ("bilinear", "nearest", "cubic")
	if type == "bilinear":
		interpolationType = Interpolation.LINEAR
	elif type == "nearest":
		interpolationType = Interpolation.NEAREST
	else:
		interpolationType = Interpolation.CUBIC

	return resizeFn(data, height, width, interpolationType)