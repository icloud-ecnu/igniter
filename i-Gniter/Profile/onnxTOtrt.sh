# Emulate "maxBatchSize" behavior from implicit batch engines by setting
# an optimization profile with min=(1, *shape), opt=max=(maxBatchSize, *shape)
MAX_BATCH_SIZE=32
INPUT_NAME="actual_input_1"

# Convert dynamic batch ONNX model to TRT Engine with optimization profile defined
#   --minShapes: kMIN shape
#   --optShapes: kOPT shape
#   --maxShapes: kMAX shape
#   --shapes:    # Inference shape - this is like context.set_binding_shape(0, shape)
trtexec --onnx=alexnet_dynamic.onnx \
        --explicitBatch \
        --minShapes=${INPUT_NAME}:1x3x224x224 \
        --optShapes=${INPUT_NAME}:${MAX_BATCH_SIZE}x3x224x224 \
        --maxShapes=${INPUT_NAME}:${MAX_BATCH_SIZE}x3x224x224 \
        --shapes=${INPUT_NAME}:1x3x224x224 \
        --saveEngine=alexnet_1_1_64.engine

trtexec --onnx=resnet50_dynamic.onnx \
        --explicitBatch \
        --minShapes=${INPUT_NAME}:1x3x224x224 \
        --optShapes=${INPUT_NAME}:${MAX_BATCH_SIZE}x3x224x224 \
        --maxShapes=${INPUT_NAME}:${MAX_BATCH_SIZE}x3x224x224 \
        --shapes=${INPUT_NAME}:1x3x224x224 \
        --saveEngine=resnet50_1_1_64.engine

trtexec --onnx=vgg19_dynamic.onnx \
        --explicitBatch \
        --minShapes=${INPUT_NAME}:1x3x224x224 \
        --optShapes=${INPUT_NAME}:${MAX_BATCH_SIZE}x3x224x224 \
        --maxShapes=${INPUT_NAME}:${MAX_BATCH_SIZE}x3x224x224 \
        --shapes=${INPUT_NAME}:1x3x224x224 \
        --saveEngine=vgg19_1_1_64.engine

trtexec --onnx=ssd_dynamic.onnx \
        --explicitBatch \
        --minShapes=${INPUT_NAME}:1x3x300x300 \
        --optShapes=${INPUT_NAME}:${MAX_BATCH_SIZE}x3x300x300 \
        --maxShapes=${INPUT_NAME}:${MAX_BATCH_SIZE}x3x300x300 \
        --shapes=${INPUT_NAME}:1x3x300x300 \
        --saveEngine=ssd_1_1_64.engine