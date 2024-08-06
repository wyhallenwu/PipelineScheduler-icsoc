#include "controller.h"
#include <cassert>

PipelineModelListType Controller::getModelsByPipelineTypeTest(PipelineType type, const std::string &startDevice, const std::string &pipelineName, const std::string &streamName) {
    std::string sourceName = streamName;
    if (ctrl_initialRequestRates.find(sourceName) == ctrl_initialRequestRates.end()) {
        for (auto [key, rates]: ctrl_initialRequestRates) {
            if (key.find(pipelineName) != std::string::npos) {
                sourceName = key;
                break;
            }
        }
    }
    switch (type) {
        case PipelineType::Traffic: {
            auto *datasource = new PipelineModel{startDevice, "datasource", {}, true, {}, {}};
            datasource->possibleDevices = {startDevice};

            auto *yolov5n = new PipelineModel{
                    "server",
                    "yolov5n",
                    {},
                    true,
                    {},
                    {},
                    {},
                    {{datasource, -1}},
                    1
            };
            yolov5n->possibleDevices = {"server"};
            datasource->downstreams.push_back({yolov5n, -1});

            // jlf added
            auto *yolov5n320 = new PipelineModel{
                    "server",
                    "yolov5n320",
                    {},
                    true,
                    {},
                    {},
                    {},
                    {{datasource, -1}},
                    1
            };
            yolov5n320->possibleDevices = {"server"};


            auto *yolov5n512 = new PipelineModel{
                    "server",
                    "yolov5n512",
                    {},
                    true,
                    {},
                    {},
                    {},
                    {{datasource, -1}},
                    1
            };
            yolov5n512->possibleDevices = {"server"};

            auto *yolov5s= new PipelineModel{
                    "server",
                    "yolov5s",
                    {},
                    true,
                    {},
                    {},
                    {},
                    {{datasource, -1}},
                    1
            };
            yolov5s->possibleDevices = {"server"};

        //     auto *yolov5m= new PipelineModel{
        //             "server",
        //             "yolov5m",
        //             {},
        //             true,
        //             {},
        //             {},
        //             {},
        //             {{datasource, -1}}
        //     };
        //     yolov5m->possibleDevices = {"server"};

            auto *retina1face = new PipelineModel{
                    "server",
                    "retina1face",
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{yolov5n, 0}, {yolov5n320, 0}, {yolov5n512, 0}, {yolov5s, 0}}
            };
            retina1face->possibleDevices = {"server"};
            yolov5n->downstreams.push_back({retina1face, 0});
            yolov5n320->downstreams.push_back({retina1face, 0});
            yolov5n512->downstreams.push_back({retina1face, 0});
            yolov5s->downstreams.push_back({retina1face, 0});
        //     yolov5m->downstreams.push_back({retina1face, 0});


            auto *arcface = new PipelineModel{
                    "server",
                    "arcface",
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{retina1face, -1}}
            };
            arcface->possibleDevices = {"server"};
            retina1face->downstreams.push_back({arcface, -1});

            auto *carbrand = new PipelineModel{
                    "server",
                    "carbrand",
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{yolov5n, 2}, {yolov5n320, 2}, {yolov5n512, 2}, {yolov5s, 2}}
            };
            carbrand->possibleDevices = {"server"};
            yolov5n->downstreams.push_back({carbrand, 2});
            yolov5n320->downstreams.push_back({carbrand, 2});
            yolov5n512->downstreams.push_back({carbrand, 2});
            yolov5s->downstreams.push_back({carbrand, 2});
        //     yolov5m->downstreams.push_back({carbrand, 2});

            auto *platedet = new PipelineModel{
                    "server",
                    "platedet",
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{yolov5n, 2}, {yolov5n320, 2}, {yolov5n512, 2}, {yolov5s, 2}}
            };
            platedet->possibleDevices = {"server"};
            yolov5n->downstreams.push_back({platedet, 2});
            yolov5n320->downstreams.push_back({platedet, 2});
            yolov5n512->downstreams.push_back({platedet, 2});
            yolov5s->downstreams.push_back({platedet, 2});
        //     yolov5m->downstreams.push_back({platedet, 2});

            auto *sink = new PipelineModel{
                    "server",
                    "sink",
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{arcface, -1}, {carbrand, -1}, {platedet, -1}}
            };
            sink->possibleDevices = {"sink"};
            arcface->downstreams.push_back({sink, -1});
            carbrand->downstreams.push_back({sink, -1});
            platedet->downstreams.push_back({sink, -1});

            if (!sourceName.empty()) {
                yolov5n->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][yolov5n->name];
                // jlf added
                // yolov5n->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][yolov5n->name];
                // yolov5n320->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][yolov5n320->name];
                // yolov5n512->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][yolov5n512->name];
                // yolov5s->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][yolov5s->name];
                // yolov5m->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][yolov5m->name];

                retina1face->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][retina1face->name];
                arcface->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][arcface->name];
                carbrand->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][carbrand->name];
                platedet->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][platedet->name];
            }

            return {datasource, yolov5n, yolov5n320, yolov5n512, yolov5s, retina1face, arcface, carbrand, platedet, sink};
        //     return {datasource, yolov5n, retina1face, arcface, carbrand, platedet, sink};
        }
        case PipelineType::Building_Security: {
            auto *datasource = new PipelineModel{startDevice, "datasource", {}, true, {}, {}};
            datasource->possibleDevices = {startDevice};
            auto *yolov5n = new PipelineModel{
                    "server",
                    "yolov5n",
                    {},
                    true,
                    {},
                    {},
                    {},
                    {{datasource, -1}},
                    1
            };
            yolov5n->possibleDevices = {"server"};
            datasource->downstreams.push_back({yolov5n, -1});

            // jlf added
            auto *yolov5n320 = new PipelineModel{
                    "server",
                    "yolov5n320",
                    {},
                    true,
                    {},
                    {},
                    {},
                    {{datasource, -1}},
                    1
            };
            yolov5n320->possibleDevices = {"server"};

            auto *yolov5n512 = new PipelineModel{
                    "server",
                    "yolov5n512",
                    {},
                    true,
                    {},
                    {},
                    {},
                    {{datasource, -1}},
                    1
            };
            yolov5n512->possibleDevices = {"server"};

            auto *yolov5s= new PipelineModel{
                    "server",
                    "yolov5s",
                    {},
                    true,
                    {},
                    {},
                    {},
                    {{datasource, -1}},
                    1
            };
            yolov5s->possibleDevices = {"server"};

        //     auto *yolov5m= new PipelineModel{
        //             "server",
        //             "yolov5m",
        //             {},
        //             true,
        //             {},
        //             {},
        //             {},
        //             {{datasource, -1}}
        //     };
        //     yolov5m->possibleDevices = {"server"};

            auto *retina1face = new PipelineModel{
                    "server",
                    "retina1face",
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{yolov5n, 0}, {yolov5n320, 0}, {yolov5n512, 0}, {yolov5s, 0}}
            };
            retina1face->possibleDevices = {"server"};
            yolov5n->downstreams.push_back({retina1face, 0});
            yolov5n320->downstreams.push_back({retina1face, 0});
            yolov5n512->downstreams.push_back({retina1face, 0});
            yolov5s->downstreams.push_back({retina1face, 0});
        //     yolov5m->downstreams.push_back({retina1face, 0});

            auto *movenet = new PipelineModel{
                    "server",
                    "movenet",
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{yolov5n, 0}, {yolov5n320, 0}, {yolov5n512, 0}, {yolov5s, 0}}
            };
            movenet->possibleDevices = {"server"};
            yolov5n->downstreams.push_back({movenet, 0});
            yolov5n320->downstreams.push_back({movenet, 0});
            yolov5n512->downstreams.push_back({movenet, 0});
            yolov5s->downstreams.push_back({movenet, 0});
        //     yolov5m->downstreams.push_back({movenet, 0});

            auto *gender = new PipelineModel{
                    "server",
                    "gender",
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{retina1face, -1}}
            };
            gender->possibleDevices = {"server"};
            retina1face->downstreams.push_back({gender, -1});

            auto *age = new PipelineModel{
                    "server",
                    "age",
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{retina1face, -1}}
            };
            age->possibleDevices = {"server"};
            retina1face->downstreams.push_back({age, -1});

            auto *sink = new PipelineModel{
                    "server",
                    "sink",
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{gender, -1}, {age, -1}, {movenet, -1}}
            };
            sink->possibleDevices = {"sink"};
            gender->downstreams.push_back({sink, -1});
            age->downstreams.push_back({sink, -1});
            movenet->downstreams.push_back({sink, -1});

            if (!sourceName.empty()) {
                yolov5n->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][yolov5n->name];
                retina1face->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][retina1face->name];
                movenet->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][movenet->name];
                gender->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][movenet->name];
                age->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][age->name];
            }

            return {datasource, yolov5n, yolov5n320, yolov5n512, yolov5s, retina1face, movenet, gender, age, sink};
        //     return {datasource, yolov5n, retina1face, movenet, gender, age, sink};
        }
        case PipelineType::Video_Call: {
            auto *datasource = new PipelineModel{startDevice, "datasource", {}, true, {}, {}};
            datasource->possibleDevices = {startDevice};
            auto *retina1face = new PipelineModel{
                    "server",
                    "retina1face",
                    {},
                    true,
                    {},
                    {},
                    {},
                    {{datasource, -1}}
            };
            retina1face->possibleDevices = {"server"};
            datasource->downstreams.push_back({retina1face, -1});

            auto *emotionnet = new PipelineModel{
                    "server",
                    "emotionnet",
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{retina1face, -1}}
            };
            emotionnet->possibleDevices = {"server"};
            retina1face->downstreams.push_back({emotionnet, -1});

            auto *age = new PipelineModel{
                    "server",
                    "age",
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{retina1face, -1}}
            };
            age->possibleDevices = {"server"};
            retina1face->downstreams.push_back({age, -1});

            auto *gender = new PipelineModel{
                    "server",
                    "gender",
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{retina1face, -1}}
            };
            gender->possibleDevices = {"server"};
            retina1face->downstreams.push_back({gender, -1});

            auto *arcface = new PipelineModel{
                    "server",
                    "arcface",
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{retina1face, -1}}
            };
            arcface->possibleDevices = {"server"};
            retina1face->downstreams.push_back({arcface, -1});

            auto *sink = new PipelineModel{
                    "server",
                    "sink",
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{emotionnet, -1}, {age, -1}, {gender, -1}, {arcface, -1}}
            };
            sink->possibleDevices = {"sink"};
            emotionnet->downstreams.push_back({sink, -1});
            age->downstreams.push_back({sink, -1});
            gender->downstreams.push_back({sink, -1});
            arcface->downstreams.push_back({sink, -1});

            if (!sourceName.empty()) {         
                retina1face->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][retina1face->name];
                emotionnet->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][emotionnet->name];
                age->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][age->name];
                gender->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][gender->name];
                arcface->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][arcface->name];
            }

            return {datasource, retina1face, emotionnet, age, gender, arcface, sink};
        }
        default:
            return {};
    }
}