from tqdm import tqdm

def writeVideoImageIo(file, path, fps, quiet):
	import imageio
	writer = imageio.get_writer(path, fps=fps)
	N = len(file)
	Range = range(N)
	if not quiet:
		Range = tqdm(Range)

	for i in Range:
		writer.append_data(file[i])
	writer.close()

def tryWriteVideo(file, path, fps, count=5, imgLib="imageio", quiet=True):
	assert imgLib in ("imageio", )
	f = {
		"imageio" : writeVideoImageIo
	}[imgLib]

	i = 0
	while True:
		try:
			return f(file, path, fps, quiet)
		except Exception as e:
			print("Path: %s. Exception: %s" % (path, e))
			i += 1

			if i == count:
				raise Exception
