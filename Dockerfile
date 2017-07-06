FROM ubuntu:16.04

RUN apt-get update && apt-get upgrade -y
RUN apt-get -y install make cmake gcc g++
RUN apt-get -y install wget unzip

#OpenCVのインストール
RUN apt-get -y install build-essential pkg-config libjpeg-dev libpng12-dev libtiff5-dev libopenexr-dev libavcodec-dev libavformat-dev yasm libfaac-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev libswscale-dev libjasper-dev libdc1394-22-dev libv4l-dev libgstreamer1.0-dev libgstreamer-plugins-base0.10-dev libgstreamer-plugins-base1.0-dev libtbb2 libtbb-dev libeigen3-dev
RUN ln -s /usr/include/libv4l1-videodev.h /usr/include/linux/videodev.h

RUN mkdir ~/tmp
RUN cd ~/tmp && wget https://github.com/Itseez/opencv/archive/3.1.0.zip && unzip 3.1.0.zip
RUN cd ~/tmp/opencv-3.1.0 && cmake CMakeLists.txt -DWITH_TBB=ON -DINSTALL_CREATE_DISTRIB=ON -DWITH_FFMPEG=OFF -DCMAKE_INSTALL_PREFIX=/usr/local
RUN cd ~/tmp/opencv-3.1.0 && make -j2 && make install
