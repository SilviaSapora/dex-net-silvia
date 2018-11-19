TARGET=dex-net

build:
	docker build . -t $(TARGET)

test:
	./run_docker.sh

run:
	./run_docker.sh "/bin/bash"

all:
	make build && make run

clean:
	docker rmi -f $(TARGET)
