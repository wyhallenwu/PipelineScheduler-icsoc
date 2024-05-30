# Builder Image
FROM pipeline-scheduler-agx as builder

#USER root
WORKDIR /home/soulsaver/PipelineScheduler/build
#RUN apt install libspdlog-dev libpqxx-dev -y
#RUN mkdir /app/build -p
#COPY ./cmake /app/cmake
#COPY ./libs /app/libs
#COPY ./src /app/src
#COPY ./CMakeLists.txt /app/CMakeLists.txt
#WORKDIR /app/build
#RUN cmake ..
#RUN make -j 8
#COPY ./jsons /app/jsons
