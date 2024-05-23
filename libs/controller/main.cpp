#include "controller.h"

int main()
{
    auto controller = new Controller();
    std::thread receiver_thread(&Controller::HandleRecvRpcs, controller);
    receiver_thread.detach();
    // =============================== added ============================

    // // periodically update the scheduling
    // // according to jellyfish paper, time interval is set to be 0.5s
    // std::thread update_thread(&Controller::update_and_adjust, controller);
    // update_thread.detach();

    // test dynamic programming: mapClient(ClientProfiles client_profile,
    // ModelProfiles model_profiles)
    // TODO: (1) figure out what is the inference time(ms or ns), fix the
    // computation of throughput (2) fix the update_and_adjust thread (3) correct
    // the scheduling algorithm (4) test with the adjust function and rpc call

    ClientProfiles client_profile;
    ModelProfiles model_profiles;
//    client_profile.add(std::string("1"), 1500, 10);
//    client_profile.add(std::string("2"), 1700, 30);
//    client_profile.add(std::string("3"), 1600, 20);
//    client_profile.add(std::string("4"), 2000, 30);
//    client_profile.add(std::string("5"), 1800, 20);

    // model_profiles.add(ModelInfo(1, 0.39, 320, 320, "yolov5n", 0.2));
    // model_profiles.add(ModelInfo(2, 2.89, 320, 320, "yolov5n", 0.2));
    // model_profiles.add(ModelInfo(4, 15.59, 320, 320, "yolov5n", 0.2));
    // model_profiles.add(ModelInfo(8, 78.79, 320, 320, "yolov5n", 0.2));
    // model_profiles.add(ModelInfo(16, 420.24, 320, 320, "yolov5n", 0.2));
    // model_profiles.add(ModelInfo(32, 2456.73, 320, 320, "yolov5n", 0.2));
    // model_profiles.add(ModelInfo(64, 16052.00, 320, 320, "yolov5n", 0.2));
    // model_profiles.add(ModelInfo(1, 0.43, 640, 640, "yolov5n", 0.4));
    // model_profiles.add(ModelInfo(2, 3.08, 640, 640, "yolov5n", 0.4));
    // model_profiles.add(ModelInfo(4, 16.11, 640, 640, "yolov5n", 0.4));
    // model_profiles.add(ModelInfo(8, 81.37, 640, 640, "yolov5n", 0.4));
    // model_profiles.add(ModelInfo(16, 418.64, 640, 640, "yolov5n", 0.4));
    // model_profiles.add(ModelInfo(32, 2451.59, 640, 640, "yolov5n", 0.4));
    // model_profiles.add(ModelInfo(64, 16197.56, 640, 640, "yolov5n", 0.4));

    model_profiles.add(ModelInfo(2, 33.3, 320, 320, "yolov5n", 0.2));
    model_profiles.add(ModelInfo(3, 37.5, 320, 320, "yolov5n", 0.2));
    model_profiles.add(ModelInfo(2, 33.3, 640, 640, "yolov5n", 0.4));
    model_profiles.add(ModelInfo(4, 40.0, 640, 640, "yolov5n", 0.4));
    auto mapping = mapClient(client_profile, model_profiles);
    std::cout << "mapping relation" << std::endl;
    for (auto &t : mapping)
    {
        std::cout << "======================" << std::endl;
        auto [model_info, clients, batch_size] = t;
        std::cout << std::get<0>(model_info) << " " << std::get<1>(model_info)
                  << " " << batch_size << std::endl;
        for (auto &client : clients)
        {
            std::cout << client.ip << " " << client.req_rate << " " << client.budget
                      << std::endl;
        }
        std::cout << "======================" << std::endl;
    }

    // ==================================================================

    std::ifstream file("../jsons/experiment.json");
    std::vector<TaskDescription::TaskStruct> tasks = json::parse(file);
    std::string command;

    while (controller->isRunning())
    {
        TaskDescription::TaskStruct task;
        std::cout << "Enter command {init, traffic, video_call, people, exit): ";
        std::cin >> command;
        if (command == "exit")
        {
            controller->Stop();
            break;
        }
        else if (command == "init")
        {
            for (auto &t : tasks)
            {
                controller->AddTask(t);
            }
            continue;
        }
        else if (command == "traffic")
        {
            task.type = PipelineType::Traffic;
        }
        else if (command == "video_call")
        {
            task.type = PipelineType::Video_Call;
        }
        else if (command == "people")
        {
            task.type = PipelineType::Building_Security;
        }
        else
        {
            std::cout << "Invalid command" << std::endl;
            continue;
        }
        std::cout << "Enter name of task(eg. yolov5n_320_640_32..): ";
        std::cin >> task.name;
        std::cout << "Enter SLO in ns: ";
        std::cin >> task.slo;
        std::cout << "Enter total path to source file: ";
        std::cin >> task.source;
        std::cout << "Enter name of source device: ";
        std::cin >> task.device;
        controller->AddTask(task);
    }
    delete controller;
    return 0;
}
