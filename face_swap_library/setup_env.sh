#!/usr/bin/env bash

pip3 install virtualenv
virtualenv ~/.virtualenv_test/faceswap_demo -p python3
source ~/.virtualenv_test/faceswap_demo/bin/activate

pip3 install pillow
pip3 install numpy
pip3 install opencv-python
pip3 install mediapipe
pip3 install matplotlib