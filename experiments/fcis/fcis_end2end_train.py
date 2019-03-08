import os
import sys

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'

# os.environ['MXNET_GPU_MEM_POOL_RESERVE'] = '90'

this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, '..', '..', 'fcis'))

import train_end2end

if __name__ == "__main__":
    train_end2end.main()
