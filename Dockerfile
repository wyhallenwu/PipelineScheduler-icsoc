FROM nvcr.io/nvidia/tensorrt:24.01-py3

COPY ./cmake-build-debug/ContainerAgent /ContainerAgent
COPY test.mp4 /test.mp4
RUN chmod +x /ContainerAgent

ENTRYPOINT ["/ContainerAgent"]