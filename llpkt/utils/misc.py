import time
import logging

strBold = lambda skk: "\033[1m {}\033[00m".format(skk)
strBlue = lambda skk: "\033[34m {}\033[00m".format(skk)
strRed = lambda skk: "\033[91m {}\033[00m".format(skk)
strGreen = lambda skk: "\033[92m {}\033[00m".format(skk)
strYellow = lambda skk: "\033[93m {}\033[00m".format(skk)
strLightPurple = lambda skk: "\033[94m {}\033[00m".format(skk)
strPurple = lambda skk: "\033[95m {}\033[00m".format(skk)
strCyan = lambda skk: "\033[96m {}\033[00m".format(skk)
strLightGray = lambda skk: "\033[97m {}\033[00m".format(skk)
strBlack = lambda skk: "\033[98m {}\033[00m".format(skk)

prBold = lambda skk: print("\033[1m {}\033[00m".format(skk))
prBlue = lambda skk: print("\033[34m {}\033[00m".format(skk))
prRed = lambda skk: print("\033[91m {}\033[00m".format(skk))
prGreen = lambda skk: print("\033[92m {}\033[00m".format(skk))
prYellow = lambda skk: print("\033[93m {}\033[00m".format(skk))
prLightPurple = lambda skk: print("\033[94m {}\033[00m".format(skk))
prPurple = lambda skk: print("\033[95m {}\033[00m".format(skk))
prCyan = lambda skk: print("\033[96m {}\033[00m".format(skk))
prLightGray = lambda skk: print("\033[97m {}\033[00m".format(skk))
prBlack = lambda skk: print("\033[98m {}\033[00m".format(skk))


def timeit(f):
    """ Decorator to time Any Function """

    def timed(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        seconds = end_time - start_time
        logging.getLogger("Timer").info(
            "   [-] %s : %2.5f sec, which is %2.5f min, which is %2.5f hour" %
            (f.__name__, seconds, seconds / 60, seconds / 3600))
        return result

    return timed


def print_cuda_statistics():
    logger = logging.getLogger("Cuda Statistics")
    import sys
    from subprocess import call
    import torch
    logger.info('__Python VERSION:  {}'.format(sys.version))
    logger.info('__pyTorch VERSION:  {}'.format(torch.__version__))
    logger.info('__CUDA VERSION')
    # call(["nvcc", "--version"])
    logger.info('__CUDNN VERSION:  {}'.format(torch.backends.cudnn.version()))
    logger.info('__Number CUDA Devices:  {}'.format(torch.cuda.device_count()))
    logger.info('__Devices')
    call(["nvidia-smi", "--format=csv",
          "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    logger.info('Active CUDA Device: GPU {}'.format(torch.cuda.current_device()))
    logger.info('Available devices  {}'.format(torch.cuda.device_count()))
    logger.info('Current cuda device  {}'.format(torch.cuda.current_device()))
