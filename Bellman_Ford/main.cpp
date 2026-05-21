#include "graph.h"

#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <algorithm>

int main()
{
    // =========================
    // LOAD GRAPH
    // =========================

    Graph graph =
        loadGraph("data/graph.txt");

    std::cout
        << "=========================\n";

    std::cout
        << "CUDA BELLMAN-FORD\n";

    std::cout
        << "=========================\n";

    std::cout
        << "Graph loaded successfully!\n";

    std::cout
        << "Vertices : "
        << graph.V
        << "\n";

    std::cout
        << "Edges    : "
        << graph.E
        << "\n";

    // =========================
    // SOURCE VERTEX
    // =========================

    int source = 0;

    // =========================
    // VISUALIZATION MODE
    // =========================

    bool enableVisualization =
        (graph.V <= 100);

    std::vector<RelaxationStep> steps;

    if (enableVisualization)
    {
        steps.reserve(100000);
    }

    // =========================
    // CPU BELLMAN-FORD
    // =========================

    std::cout
        << "\nRunning CPU Bellman-Ford...\n";

    auto cpuStart =
        std::chrono::high_resolution_clock::now();

    std::vector<int> cpuDist =
        bellmanFordCPU(
            graph,
            source,
            steps);

    auto cpuEnd =
        std::chrono::high_resolution_clock::now();

    double cpuTime =
        std::chrono::duration<double, std::milli>(
            cpuEnd - cpuStart).count();

    // =========================
    // SAVE RELAXATION STEPS
    // =========================

    if (enableVisualization)
    {
        std::ofstream out(
            "results/steps.txt");

        if (out.is_open())
        {
            for (const auto& s : steps)
            {
                out
                    << s.iteration << " "
                    << s.u << " "
                    << s.v << " "
                    << s.oldDist << " "
                    << s.newDist << " "
                    << s.updated
                    << "\n";
            }

            out.close();

            std::cout
                << "Visualization steps exported.\n";
        }
    }
    else
    {
        std::cout
            << "Skipping visualization export for large graph.\n";
    }

    // =========================
    // GPU BELLMAN-FORD
    // =========================

    std::cout
        << "\nRunning CUDA Bellman-Ford...\n";

    auto gpuStart =
        std::chrono::high_resolution_clock::now();

    std::vector<int> gpuDist =
        bellmanFordGPU(
            graph,
            source);

    auto gpuEnd =
        std::chrono::high_resolution_clock::now();

    double gpuTime =
        std::chrono::duration<double, std::milli>(
            gpuEnd - gpuStart).count();

    // =========================
    // VERIFY CORRECTNESS
    // =========================

    bool correct = true;

    for (int i = 0; i < graph.V; i++)
    {
        if (cpuDist[i] != gpuDist[i])
        {
            correct = false;

            std::cout
                << "\nMismatch at vertex "
                << i
                << "\n";

            std::cout
                << "CPU : "
                << cpuDist[i]
                << "\n";

            std::cout
                << "GPU : "
                << gpuDist[i]
                << "\n";

            break;
        }
    }

    // =========================
    // SAMPLE OUTPUT
    // =========================

    std::cout
        << "\n=========================\n";

    std::cout
        << "CPU RESULT SAMPLE\n";

    std::cout
        << "=========================\n";

    for (int i = 0;
         i < std::min(10, (int)cpuDist.size());
         i++)
    {
        std::cout
            << "Vertex "
            << i
            << " : "
            << cpuDist[i]
            << "\n";
    }

    if (cpuDist.size() > 20)
    {
        std::cout
            << "...\n";

        for (int i = cpuDist.size() - 10;
             i < cpuDist.size();
             i++)
        {
            std::cout
                << "Vertex "
                << i
                << " : "
                << cpuDist[i]
                << "\n";
        }
    }

    // =========================
    // GPU SAMPLE
    // =========================

    std::cout
        << "\n=========================\n";

    std::cout
        << "GPU RESULT SAMPLE\n";

    std::cout
        << "=========================\n";

    for (int i = 0;
         i < std::min(10, (int)gpuDist.size());
         i++)
    {
        std::cout
            << "Vertex "
            << i
            << " : "
            << gpuDist[i]
            << "\n";
    }

    if (gpuDist.size() > 20)
    {
        std::cout
            << "...\n";

        for (int i = gpuDist.size() - 10;
             i < gpuDist.size();
             i++)
        {
            std::cout
                << "Vertex "
                << i
                << " : "
                << gpuDist[i]
                << "\n";
        }
    }

    // =========================
    // FINAL SUMMARY
    // =========================

    std::cout
        << "\n=========================\n";

    std::cout
        << "FINAL SUMMARY\n";

    std::cout
        << "=========================\n";

    if (correct)
    {
        std::cout
            << "CPU and GPU results MATCH!\n";
    }
    else
    {
        std::cout
            << "CPU and GPU results DO NOT MATCH!\n";
    }

    std::cout
        << "\nCPU Time : "
        << cpuTime
        << " ms\n";

    std::cout
        << "GPU Time : "
        << gpuTime
        << " ms\n";

    double speedup =
        cpuTime / gpuTime;

    std::cout
        << "Speedup : "
        << speedup
        << "x\n";

    std::cout
        << "=========================\n";

    return 0;
}

// nvcc main.cpp graph.cpp cpu_bellmanford.cpp gpu_bellmanford.cu -o app
// app.exe