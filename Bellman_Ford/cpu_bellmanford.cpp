#include "graph.h"

#include <iostream>
#include <vector>

const int INF = 1e9;

std::vector<int>
bellmanFordCPU(
    const Graph& graph,
    int source,
    std::vector<RelaxationStep>& steps)
{
    std::vector<int> dist(graph.V, INF);

    dist[source] = 0;

    // Relax edges V-1 times
    for (int i = 0; i < graph.V - 1; i++)
    {
        for (const auto& e : graph.edges)
        {
            RelaxationStep step;

            step.iteration = i + 1;

            step.u = e.src;
            step.v = e.dst;

            step.oldDist = dist[e.dst];

            step.updated = false;

            if (dist[e.src] != INF &&
                dist[e.src] + e.weight < dist[e.dst])
            {
                dist[e.dst] =
                    dist[e.src] + e.weight;

                step.updated = true;
            }

            step.newDist = dist[e.dst];

            if (steps.capacity() > 0)
            {
                steps.push_back(step);
            }
        }
    }

    return dist;
}