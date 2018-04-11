from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import download_cifar10 as download
import load_cifar10

# path to save model
path = "image-model.ckpt"

def main():
    # load data
    download.download()

    # train model
        # save

    # evaluate

    # print out save path
    print("Save to", path)

if __name__ == '__main__':
    main()