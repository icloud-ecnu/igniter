import keras
import tensorflow as tf
from keras2onnx import convert_keras

def keras_to_onnx(model, output_filename):
   onnx = convert_keras(model, output_filename)
   with open(output_filename, "wb") as f:
       f.write(onnx.SerializeToString())

semantic_model = keras.models.load_model('semantic_segmentation.hdf5')
keras_to_onnx(semantic_model, 'semantic_segmentation.onnx') 
