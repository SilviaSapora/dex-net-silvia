export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib:/usr/local/lib64:/home/silvia/dex-net/deps/meshpy/meshpy"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/freenect2/lib"
export PYTHONPATH="/usr/local/lib/python2.7/dist-packages/vtk:$LD_LIBRARY_PATH"
python tools/visualize_gqcnn_dataset.py data/datasets/gqcnn_silvia_procedural_96_96_baxter_test
