#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <random>
#include <fstream>
#include <vector>
#include <cmath>

class Stopwatch
{
public:
    Stopwatch() {}

    void start()
    {
        start_time = std::chrono::high_resolution_clock::now();
    }

    void stop()
    {
        end_time = std::chrono::high_resolution_clock::now();
    }

    double elapsed_seconds()
    {
        return std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_time;
};

int main()
{
    const int height = 1280;                                           // Height of the input tensor
    const int width = 720;                                            // Width of the input tensor
    const int numRuns = 1280;                                          // Number of valid timing runs
    const int warmupRuns = 720;                                        // Number of warmup runs


    int deviceId = 2; // Change this to the ID of the GPU you want to use
    cudaError_t err = cudaSetDevice(deviceId);
    if (err != cudaSuccess) {
        std::cerr << "Failed to set device: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize a random tensor
    std::random_device rd;                       // Random device for seeding the random number generator
    std::mt19937 gen(rd());                      // Mersenne Twister pseudo-random number generator
    std::uniform_int_distribution<> dis(0, 255); // Uniform distribution between 0 and 255 (for random pixel values)

    // Open CSV files for writing upload and download times
    std::ofstream uploadTimeFile("upload_times.csv");
    std::ofstream downloadTimeFile("download_times.csv");

    // Write CSV headers
    uploadTimeFile << "MemSize,Time (microseconds)\n";
    downloadTimeFile << "MemSize,Time (microseconds)\n";


    // Generate fractions from 0.1 to 0.4 with a step size of 0.03
    std::vector<double> fractions;
    for (double fraction = 0.2; fraction <= 0.7; fraction += 0.0001)
    {
        fractions.push_back(fraction);
        float fraction_height = height * fraction;
        float fraction_width = width * fraction;
        cv::Mat inputTensor(ceil(fraction_height), ceil(fraction_width), CV_8UC3); // Create an empty input tensor
        // Fill the input tensor with random pixel values
        for (int row = 0; row < fraction_height; ++row)
        {
            for (int col = 0; col < fraction_width; ++col)
            {
                inputTensor.at<cv::Vec3b>(row, col) = cv::Vec3b(dis(gen), dis(gen), dis(gen));
            }
        }
        Stopwatch stopwatch; // Create a Stopwatch object for timing measurements
        
        cv::Mat cpuTensor; // Create a temporary Mat for downloading the tensor back to CPU memory
        // Warmup phase: perform a few iterations of upload and download to stabilize timing
        for (int i = 0; i < warmupRuns; ++i)
        {
            cv::cuda::GpuMat gpuTensor(inputTensor); // Create a temporary GpuMat from the input tensor
            gpuTensor.download(cpuTensor); // Download gpuTensor back to CPU memory
        }

        std::vector<int64_t> uploadTimes, downloadTimes; // Vectors to store timing measurements

        // Perform numRuns iterations of upload and download for the current fraction
        for (int i = 0; i < numRuns; ++i)
        {
            stopwatch.start();
            cv::cuda::GpuMat gpuTensor(inputTensor); //UPLOAD
            stopwatch.stop();
            uploadTimes.push_back(stopwatch.elapsed_seconds()); // Store upload time

            stopwatch.start();
            gpuTensor.download(cpuTensor); // DOWNLOAD
            stopwatch.stop();
            downloadTimes.push_back(stopwatch.elapsed_seconds()); // Store download time
        }

        // Calculate the average upload and download times
        double avgUploadTime = std::accumulate(uploadTimes.begin(), uploadTimes.end(), 0ll) / uploadTimes.size();
        double avgDownloadTime = std::accumulate(downloadTimes.begin(), downloadTimes.end(), 0ll) / downloadTimes.size();

        // Write the average times to the CSV files
        int memSize = fraction_height * fraction_width * 3 * 4; // Size of the fraction tensor in bytes
        uploadTimeFile << memSize << "," << avgUploadTime << "\n";
        downloadTimeFile << memSize << "," << avgDownloadTime << "\n";

        // Print the results to the console
        std::cout << "Transferred " << memSize << " bytes of tensor. ";
        std::cout << "Upload time: " << avgUploadTime << " μs | ";
        std::cout << "Download time: " << avgDownloadTime << " μs" << std::endl;

    }


    // Close the CSV files
    uploadTimeFile.close();
    downloadTimeFile.close();

    return 0;
}