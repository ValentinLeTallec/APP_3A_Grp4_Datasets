import json, cv2, tarfile, os, tempfile
# from pygit import Repo
from six.moves import urllib
import numpy as np
from PIL import Image

# %tensorflow_version 1.x
import tensorflow as tf

# Import Retinex from git
# try:
#     init
# except NameError:
#   git_url = 'https://github.com/dongb5/Retinex.git'
#   Repo.clone_from(git_url, './', branch='master', bare=True)
#   init = False
# import Retinex.retinex as retinex


############# Define NN model used for segmentation #############

class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map

############# Download the pretrained model #############

MODEL_NAME = 'xception_coco_voctrainval' 

DOWNLOAD_URL = 'http://download.tensorflow.org/models/deeplabv3_pascal_trainval_2018_01_04.tar.gz'
_TARBALL_NAME = 'deeplab_model.tar.gz'

model_dir = tempfile.mkdtemp()
tf.gfile.MakeDirs(model_dir)

download_path = os.path.join(model_dir, _TARBALL_NAME)
print('downloading model, this might take a while...')
urllib.request.urlretrieve(DOWNLOAD_URL, download_path)
print('download completed! loading DeepLab model...')

MODEL = DeepLabModel(download_path)
print('model loaded successfully!')

############# Define what we concidere to be the foreground #############


LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

# Les gens ont parfois tendence à être classifier comme animal mais pas comme objet
is_part_of_foreground = [
    # 'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
          0       ,     0      ,     0    ,   1   ,   0   ,    0    ,   0  ,
    # 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
       0  ,   1  ,    0   ,   1  ,       0      ,   1  ,    1   ,      0     ,
    # 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
        1    ,      0       ,    1   ,    0  ,    0   ,   0
]

############# The function we actually export #############

def removeBackground(image):

  with open('Retinex/config.json', 'r') as f:
        config = json.load(f)
  imRetinex = retinex.automatedMSRCR(image, config['sigma_list'])
  imRetinex = Image.fromarray(imRetinex)
  resized_im, seg_map = MODEL.run(imRetinex)

  segmBinaire = np.array([ [is_part_of_foreground[e] for e in f] for f in seg_map])
  segmBinaire = cv2.resize(np.float32(segmBinaire), dsize=image.size, interpolation=cv2.INTER_LINEAR)
  
  mask = np.where((segmBinaire==0),0,1).astype('uint8')
  imgNoBg = image*mask[:,:,np.newaxis]
  # imgNoBg = Image.fromarray(imgNoBg)
  return imgNoBg