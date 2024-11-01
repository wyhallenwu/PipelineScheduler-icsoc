# Builder Image
ARG JETPACK_VERSION="r35.2.1-py3"

FROM pipeline-scheduler:${JETPACK_VERSION}

USER root
RUN mkdir /app/build -p
COPY ./cmake /app/cmake
COPY ./src /app/src
COPY ./libs /app/libs
COPY ./CMakeLists.txt /app/CMakeLists.txt
WORKDIR /app/build
RUN cmake -DCMAKE_BUILD_TYPE=VERSION ..
RUN make -j 32
COPY ./jsons /app/jsons