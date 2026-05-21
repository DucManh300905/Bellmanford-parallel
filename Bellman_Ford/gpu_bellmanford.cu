#include "graph.h"

#include <cuda_runtime.h>

#include <vector>
#include <iostream>

#define INF 1000000000

// =========================
// CUDA KERNEL
// =========================

__global__
void relaxEdges(
    Edge* edges,
    int E,
    int* oldDist,
    int* newDist)
{
    int idx =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    if (idx < E)
    {
        Edge e = edges[idx];

        if (oldDist[e.src] != INF)
        {
            int newDistance =
                oldDist[e.src] + e.weight;

            atomicMin(
                &newDist[e.dst],
                newDistance);
        }
    }
}

// =========================
// GPU BELLMAN-FORD
// =========================

std::vector<int>
bellmanFordGPU(
    const Graph& graph,
    int source)
{
    int V = graph.V;
    int E = graph.E;

    // =========================
    // HOST DISTANCES
    // =========================

    std::vector<int> h_dist(
        V,
        INF);

    h_dist[source] = 0;

    // =========================
    // DEVICE MEMORY
    // =========================

    Edge* d_edges;

    int* d_oldDist;
    int* d_newDist;

    cudaMalloc(
        &d_edges,
        E * sizeof(Edge));

    cudaMalloc(
        &d_oldDist,
        V * sizeof(int));

    cudaMalloc(
        &d_newDist,
        V * sizeof(int));

    // =========================
    // COPY EDGES
    // =========================

    cudaMemcpy(
        d_edges,
        graph.edges.data(),
        E * sizeof(Edge),
        cudaMemcpyHostToDevice);

    cudaMemcpy(
        d_oldDist,
        h_dist.data(),
        V * sizeof(int),
        cudaMemcpyHostToDevice);

    // =========================
    // CUDA CONFIG
    // =========================

    int threadsPerBlock = 256;

    int blocksPerGrid =
        (E + threadsPerBlock - 1)
        / threadsPerBlock;

    // =========================
    // BELLMAN-FORD ITERATIONS
    // =========================

    for (int i = 0; i < V - 1; i++)
    {
        // Copy oldDist -> newDist

        cudaMemcpy(
            d_newDist,
            d_oldDist,
            V * sizeof(int),
            cudaMemcpyDeviceToDevice);

        // Launch kernel

        relaxEdges<<<
            blocksPerGrid,
            threadsPerBlock>>>(
                d_edges,
                E,
                d_oldDist,
                d_newDist);

        cudaDeviceSynchronize();

        // Swap buffers

        std::swap(
            d_oldDist,
            d_newDist);
    }

    // =========================
    // COPY RESULT BACK
    // =========================

    cudaMemcpy(
        h_dist.data(),
        d_oldDist,
        V * sizeof(int),
        cudaMemcpyDeviceToHost);

    // =========================
    // FREE MEMORY
    // =========================

    cudaFree(d_edges);

    cudaFree(d_oldDist);

    cudaFree(d_newDist);

    return h_dist;
}