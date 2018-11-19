IP=$(ifconfig en0 | grep inet | awk '$1=="inet" {print $2}')
xhost + $IP
docker run -it -e DISPLAY=$IP:0 -v /tmp/.X11-unix:/tmp/.X11-unix dex-net ${1:-""}
