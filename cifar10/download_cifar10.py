import tarfile
import tensorflow as tf
import os
import sys
from six.moves import urllib

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', 'cifar10_data', """Path to the CIFAR-10 data directory.""")

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

def download():
    """Download and extract the tarball from Alex's website."""
    dest_directory = FLAGS.data_dir

    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)

    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')

    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)
