TARGET=dex-net

build:
	docker build . -t $(TARGET)

run:
	xhost + 127.0.0.1 && docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro $(TARGET)

all:
	make build && make run

clean:
	docker rmi -f $(TARGET)
