import os
import h5py
import numpy as np
from argparse import ArgumentParser
from neural_wrappers.utilities import h5StoreDict

def getArgs():
    parser = ArgumentParser()
    parser.add_argument("mode")
    parser.add_argument("datasetPath")
    parser.add_argument("--exportFileName")

    args = parser.parse_args()
    assert args.mode in ("train", "test")
    if args.exportFileName is None:
        args.exportFileName = "%s.h5" % (args.mode)
    if args.mode == "test":
        assert not args.exportFileName is None
    return args

def getPaths(datasetPath, splits, expectedKeys, topLevelRenames):
    keyPaths = {key : [] for key in expectedKeys}
    for key in expectedKeys:
        expectedEnd = "png"
        if key == "depth":
            expectedEnd = "npy"
        thisItems = sorted(list(filter(lambda x : x.endswith(expectedEnd), os.listdir("%s/%s" % (datasetPath, key)))))
        keyPaths[key] = thisItems
    
    # Sanity check to see that all files are there.
    test = keyPaths["img"]
    for key in expectedKeys:
        assert len(test) == len(keyPaths[key])
        for i in range(len(keyPaths[key])):
            assert test[i].split(".")[0] == keyPaths[key][i].split(".")[0]

    # Compute start:end points for each key
    np.random.seed(42)
    perm = np.random.permutation(len(test))
    nKeys = {}
    nStart = 0
    for key in splits:
        nEnd = nStart + int(len(test) * splits[key])
        nKeys[key] = (nStart, nEnd)
        nStart = nEnd
    nKeys[key] = (nKeys[key][0], len(test))

    # Store the randomized paths according to our split
    result = {splitKey : {key : [] for key in topLevelRenames} for splitKey in splits}
    for key in splits:
        start, end = nKeys[key]
        for i in range(start, end):
            ix = perm[i]
            for keyPath, renamedKeyPath in zip(expectedKeys, topLevelRenames):
                item = keyPaths[keyPath][ix]
                result[key][renamedKeyPath].append(item)
        for renamedKeyPath in topLevelRenames:
            result[key][renamedKeyPath] = np.array(result[key][renamedKeyPath], "S")
 
    for splitKey in splits:
        print(splitKey, ["%s => %s" % (key, len(result[splitKey][key])) for key in topLevelRenames])
    return result

def main():
    args = getArgs()
    Dirs = os.listdir(args.datasetPath)
    expectedKeys = ["seg", "normal_mask", "img", "halftone", "depth"]
    topLevelRenames = ["semantic_segmentation", "normal", "rgb", "halftone", "depth"]
    assert sum([item in Dirs for item in expectedKeys]) == 5
    splits = {"train" : 0.8, "validation" : 0.2} if args.mode == "train" else {"test" : 1.0}

    file = h5py.File(args.exportFileName, "w")
    paths = getPaths(args.datasetPath, splits, expectedKeys, topLevelRenames)
    h5StoreDict(file, paths)
    h5StoreDict(file, {"others" : {
        "baseDirectory" : os.path.abspath(args.datasetPath),
        "maxDepthMeters" : 10,
        "datasetName" : "NYUDepthV2"
    }})
    file.flush()
    print("Stored h5 file to %s" % args.exportFileName)

if __name__ == "__main__":
    main()