# Builder Image
FROM pipeplusplus:dev

USER root
RUN mkdir /app/build -p
COPY ./cmake /app/cmake
COPY ./src /app/src
COPY ./libs /app/libs
COPY ./CMakeLists.txt /app/CMakeLists.txt
WORKDIR /app/build
RUN cmake ..
RUN make -j 32
COPY ./jsons /app/jsons