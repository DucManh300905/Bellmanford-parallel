#include "graph.h"

#include <iostream>
#include <fstream>

Graph loadGraph(const std::string& filename)
{
    Graph graph;

    std::ifstream file(filename);

    if (!file.is_open())
    {
        std::cerr << "Cannot open file\n";
        exit(1);
    }

    file >> graph.V >> graph.E;

    graph.edges.resize(graph.E);

    for (int i = 0; i < graph.E; i++)
    {
        file >> graph.edges[i].src
             >> graph.edges[i].dst
             >> graph.edges[i].weight;
    }

    file.close();

    return graph;
}

void printGraph(const Graph& graph)
{
    std::cout << "Vertices: "
              << graph.V << "\n";

    std::cout << "Edges: "
              << graph.E << "\n";

    for (const auto& e : graph.edges)
    {
        std::cout
            << e.src << " -> "
            << e.dst << " ("
            << e.weight << ")\n";
    }
}