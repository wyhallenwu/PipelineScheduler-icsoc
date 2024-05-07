# Builder Image
FROM cuda11.4.4_tensorrt8.4.3.1_grpc1.62:dev as builder

USER root
RUN apt install libspdlog-dev -y
RUN mkdir ${HOME}/app/build -p
COPY ./cmake ${HOME}/app/cmake
COPY ./libs ${HOME}/app/libs
COPY ./src ${HOME}/app/src
COPY ./CMakeLists.txt ${HOME}/app/CMakeLists.txt
WORKDIR ${HOME}/app/build
RUN cmake ..
RUN make -j 16
COPY ./models ${HOME}/app/models
COPY ./jsons ${HOME}/app/jsons