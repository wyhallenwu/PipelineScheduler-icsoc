# Builder Image
FROM pipeplusplus:dev

USER root
RUN apt install libspdlog-dev libpqxx-dev -y
RUN mkdir /app/build -p
COPY ./cmake /app/cmake
COPY ./src /app/src
COPY ./libs /app/libs
COPY ./CMakeLists.txt /app/CMakeLists.txt
WORKDIR /app/build
RUN cmake ..
RUN make -j 16
COPY ./jsons /app/jsons