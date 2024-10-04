# PipelineScheduler

PipelinePlusPlus (p++ or ppp) is the c++ implementation of the paper OctopInf: Resource-Efficient and Content-Aware Edge Video Analytics Real-Time Inference Serving.
When using our Code please cite the following paper:

**SPACE FOR PAPER CITATION**

## Branches

The repository has 4 main branches for the different devices used in the experiments: Edge server, Xavier AGX, Xavier NX and Orin Nano.
This is because of library conflicts and easier tracking of the configurations used during the runtime.
The sub-branches contain all changes from master, with additional device specific modifications e.g. in the profiler.

## Directory Structure

The main source code can be found within the `libs/` folder while `src/` contains the data sink and simulates the end-user receiving the data.
Configurations for models and experiments can be found in `jsons/` while the directories `cmake`, `scripts/`, and `dockerfiles` show deployment related code and helpers.
For analyzing the results we provide python scripts in `analyzes`.
The Dockerfile and CmakeList in the root directory are the main entry points to deploy the system.
