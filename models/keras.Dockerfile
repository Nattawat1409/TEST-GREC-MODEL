# TensorFlow 1.15 environment for the official Keras MCN and VLT code.
# NVIDIA's TF1 image runs on current GPUs (Ampere/Hopper), which the original TF 1.9 / 1.15 wheels do not.
#   MCN:  docker build -f models/keras.Dockerfile --build-arg KERAS_VERSION=2.2.4 -t grec-mcn .
#   VLT:  docker build -f models/keras.Dockerfile --build-arg KERAS_VERSION=2.3.1 -t grec-vlt .
FROM nvcr.io/nvidia/tensorflow:23.03-tf1-py3

ARG KERAS_VERSION=2.3.1
RUN pip install --no-cache-dir \
        "keras==${KERAS_VERSION}" "h5py==2.10.0" "spacy==2.3.9" "yacs==0.1.8" \
        "keras-transformer==0.38.0" "keras-bert==0.80.0" \
        progressbar2 matplotlib "opencv-python-headless<4.9" "pillow<10" "numpy<1.24" \
 && pip install --no-cache-dir \
        https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.3.0/en_vectors_web_lg-2.3.0.tar.gz
