export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib:/usr/local/lib64:/home/silvia/dex-net/deps/meshpy/meshpy"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/freenect2/lib"
export PYTHONPATH="/usr/local/lib/python2.7/dist-packages/vtk:$LD_LIBRARY_PATH"
python v-rep-grasping/src/gqcnn_execution.py
