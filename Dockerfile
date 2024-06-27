# Builder Image
FROM pipeline-scheduler-orin-nano

USER root
RUN pip install -U jetson-stats --force
WORKDIR /home/soulsaver/PipelineScheduler/build
#RUN apt install libspdlog-dev libpqxx-dev -y
#RUN mkdir /app/build -p
#COPY ./cmake /app/cmake
#COPY ./outputbuffer.pl /app/build/outputbuffer.pl
#RUN chmod +x /app/build/outputbuffer.pl
#COPY ./src /app/src
#COPY ./libs /app/libs
#COPY ./CMakeLists.txt /app/CMakeLists.txt
#WORKDIR /app/build
#RUN cmake ..
#RUN make -j 8
#COPY ./jsons /app/jsons
