#!/bin/sh
# ******************************************************************************
# A bash script to install Python modules in the project environment required
# by the project.
# ******************************************************************************
pip install --upgrade \
  opencv-python \
  torch \
  torchvision \
  imageio \
  imageio-ffmpeg
