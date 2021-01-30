import os
from tqdm import trange

class Debug:
    envVar = "NW_LOGLEVEL"
    logLevels = {
        "print" : 1,
        "tqdm" : 1,
        "default" : 1
    }

    @staticmethod
    def getMinLogLevel(check):
        if not check in Debug.logLevels:
            Debug.print("Warning! '%s' not in log levels. Using 'default'." % check)
            return Debug.logLevels["default"]
        return Debug.logLevels[check]

    @staticmethod
    def getLogLevel():
        assert Debug.envVar in os.environ, "%s not in environment variables." % Debug.envVar
        ret = int(os.environ[Debug.envVar])
        assert ret >= 0
        return ret

    @staticmethod
    def use(check):
        return Debug.getLogLevel() >= Debug.getMinLogLevel(check)

    @staticmethod
    def print(msg):
        return Debug.log(Debug.getMinLogLevel("print"), msg)

    @staticmethod
    def log(level, msg):
        if Debug.getLogLevel() >= level:
            print(msg)

    @staticmethod
    def range(*args, **kwargs):
        if Debug.use("tqdm"):
            return trange(*args, **kwargs)
        else:
            return range(*args)
