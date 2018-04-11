from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

path = "image-model.ckpt"

def main():
    print(path)

if __name__ == '__main__':
    main()