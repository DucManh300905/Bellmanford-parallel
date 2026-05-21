#pragma once

#include <vector>
#include <string>

struct Edge
{
    int src;
    int dst;
    int weight;
};

struct Graph
{
    int V;
    int E;

    std::vector<Edge> edges;
};

struct RelaxationStep
{
    int iteration;

    int u;
    int v;

    int oldDist;
    int newDist;

    bool updated;
};

// Graph functions
Graph loadGraph(const std::string& filename);

void printGraph(const Graph& graph);

// CPU Bellman-Ford
std::vector<int>
bellmanFordCPU(
    const Graph& graph,
    int source,
    std::vector<RelaxationStep>& steps);

// GPU Bellman-Ford
std::vector<int>
bellmanFordGPU(
    const Graph& graph,
    int source);