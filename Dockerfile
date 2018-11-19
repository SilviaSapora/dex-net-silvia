# docker build -t ubuntu1604py36
FROM ubuntu:16.04

RUN apt-get update

RUN apt-get install -y build-essential python2.7 python2.7-dev python-pip
RUN apt-get install -y git cmake sudo libvtk5-dev python-vtk python-sip python-qt4 libosmesa6-dev meshlab libhdf5-dev python-tk
RUN apt-get install -y libboost-all-dev assimp-utils libassimp-dev
RUN apt-get install -y freeglut3-dev libxmu-dev libxi-dev libopenimageio-dev mesa-utils

# update pip
RUN python2.7 -m pip install pip --upgrade
RUN python2.7 -m pip install wheel numpy vtk
ENV PYTHONPATH="/usr/local/lib/python2.7/dist-packages/vtk:${LD_LIBRARY_PATH}"

RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo

WORKDIR /app
COPY . /app

RUN sh install.sh cpu python
ENV LD_LIBRARY_PATH="/app/deps/meshpy/meshpy:/usr/local/lib64/"

RUN python2.7 -m pip install imageio shapely meshrender

CMD ["python", "setup.py", "test"]
