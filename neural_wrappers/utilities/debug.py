import os
from tqdm import trange

def dprint(msg):
    assert "NW_QUIET" in os.environ
    quiet = int(os.environ["NW_QUIET"])
    assert quiet in (0, 1)
    if quiet == 0:
        print(msg)

def drange(*args, **kwargs):
    assert "NW_QUIET" in os.environ
    quiet = int(os.environ["NW_QUIET"])
    assert quiet in (0, 1)
    if quiet == 0:
        return trange(*args, **kwargs)
    else:
        return range(*args)
