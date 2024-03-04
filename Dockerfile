# Builder Image
FROM cuda11.4.4_tensorrt8.4.3.1_grpc1.62 as builder

COPY ./libs /app/libs
COPY ./src /app/src
COPY ./CMakeLists.txt /app/CMakeLists.txt
RUN mkdir /app/build
WORKDIR /app/build
RUN cmake ..
RUN make -j 4

# Final Image
FROM cuda11.4.4_tensorrt8.4.3.1_grpc1.62
COPY --from=builder /app/build/Container* /app/
RUN chmod +x /app/Yolov5Agent /app/DataSource
COPY ./test.mp4 /app/test.mp4