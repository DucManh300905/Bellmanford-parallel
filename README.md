# Bellman-Ford: CPU vs GPU (CUDA)

Dự án so sánh hiệu năng thuật toán **Bellman-Ford tìm đường đi ngắn nhất** giữa hai phiên bản: xử lý tuần tự trên CPU và xử lý song song trên GPU bằng CUDA. Đồ thị được sinh ngẫu nhiên với quy mô lớn (100.000 đỉnh, 500.000 cạnh) để đánh giá rõ sự chênh lệch tốc độ.

---

## Cấu trúc dự án

```
Bellman_Ford/
├── data/
│   └── graph.txt              # Đồ thị đầu vào (sinh tự động)
├── results/                   # Kết quả khoảng cách ngắn nhất
├── visualization/             # Biểu đồ so sánh hiệu năng
├── generate_graph.py          # Sinh đồ thị ngẫu nhiên
├── graph.h                    # Định nghĩa cấu trúc Graph, Edge
├── graph.cpp                  # Đọc đồ thị từ file
├── cpu_bellmanford.cpp        # Bellman-Ford trên CPU
├── gpu_bellmanford.cu         # Bellman-Ford trên GPU (CUDA)
├── main.cpp                   # Entry point, đo thời gian, so sánh
├── app.exe                    # Executable
└── README.md
```

---

## Yêu cầu

| Thành phần | Yêu cầu |
|---|---|
| Compiler C++ | g++ hoặc MSVC (C++17) |
| CUDA Toolkit | >= 11.0 |
| GPU | NVIDIA với CUDA support |
| Python | >= 3.8 (để sinh đồ thị) |

---

## Cấu trúc dữ liệu

### `graph.h`

```cpp
struct Edge {
    int src, dst, weight;
};

struct Graph {
    int V, E;
    std::vector<Edge> edges;
};

struct RelaxationStep {
    int iteration;
    int u, v;
    int oldDist, newDist;
    bool updated;
};
```

`RelaxationStep` được dùng riêng ở phiên bản CPU để ghi lại từng bước relax — phục vụ visualization và debug.

---

## Bước 1 — Sinh đồ thị (`generate_graph.py`)

Sinh đồ thị có hướng ngẫu nhiên với trọng số dương và lưu ra file.

| Tham số | Giá trị |
|---|---|
| Số đỉnh (`V`) | 100.000 |
| Số cạnh (`E`) | 500.000 |
| Trọng số (`w`) | 1 → 20 (ngẫu nhiên) |
| Output | `data/graph.txt` |

**Định dạng file:**
```
V E
u1 v1 w1
u2 v2 w2
...
```

Dùng `set` để đảm bảo không có cạnh trùng lặp. Bỏ qua self-loop (`u == v`).

```bash
python generate_graph.py
```

---

## Bước 2 — Bellman-Ford CPU (`cpu_bellmanford.cpp`)

**Thuật toán:** Bellman-Ford tuần tự (Sequential)

### Nguyên lý

Thực hiện **V - 1 vòng lặp**, mỗi vòng duyệt qua toàn bộ E cạnh và relax nếu tìm được đường ngắn hơn:

$$d[v] = \min(d[v],\ d[u] + w(u, v))$$

### Pseudocode

```
dist[source] = 0
dist[v] = INF  (với mọi v ≠ source)

for i in 1 .. V-1:
    for each edge (u, v, w):
        if dist[u] + w < dist[v]:
            dist[v] = dist[u] + w
```

### Ghi lại bước relax

Mỗi lần xét cạnh, CPU ghi một `RelaxationStep` vào vector `steps`:

```cpp
RelaxationStep step;
step.iteration = i + 1;
step.u = e.src;
step.v = e.dst;
step.oldDist = dist[e.dst];
step.updated = (dist[e.src] + e.weight < dist[e.dst]);
step.newDist = dist[e.dst];
steps.push_back(step);
```

Dữ liệu này được dùng để vẽ biểu đồ quá trình hội tụ.

### Độ phức tạp

| | CPU |
|---|---|
| Thời gian | O(V × E) |
| Không gian | O(V + E) |

---

## Bước 3 — Bellman-Ford GPU (`gpu_bellmanford.cu`)

**Thuật toán:** Bellman-Ford song song trên CUDA

### Ý tưởng song song hóa

Ở mỗi vòng lặp, CPU phải duyệt E cạnh **tuần tự**. Với GPU, toàn bộ E cạnh được relax **đồng thời** — mỗi CUDA thread xử lý một cạnh.

### CUDA Kernel

```cuda
__global__ void relaxEdges(Edge* edges, int E, int* oldDist, int* newDist)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < E) {
        Edge e = edges[idx];
        if (oldDist[e.src] != INF) {
            int newDistance = oldDist[e.src] + e.weight;
            atomicMin(&newDist[e.dst], newDistance);
        }
    }
}
```

**`atomicMin`**: Đảm bảo tính đúng đắn khi nhiều thread cùng cập nhật một đỉnh đích, tránh race condition.

### Pipeline GPU mỗi vòng lặp

```
[Host] Copy oldDist → newDist  (cudaMemcpy D2D)
         ↓
[GPU]  relaxEdges<<<blocks, 256>>>()   ← E thread song song
         ↓
[Host] cudaDeviceSynchronize()
         ↓
[Host] swap(d_oldDist, d_newDist)
```

### Cấu hình CUDA

```cpp
int threadsPerBlock = 256;
int blocksPerGrid   = (E + 255) / 256;
```

Với E = 500.000 cạnh → ~1954 blocks × 256 threads.

### Quản lý bộ nhớ

```
Host (RAM)                    Device (VRAM)
──────────────────────────    ──────────────────────
h_dist[V]          →  cudaMemcpy  →  d_oldDist[V]
graph.edges[E]     →  cudaMemcpy  →  d_edges[E]
                                     d_newDist[V]
                   ←  cudaMemcpy  ←
h_dist[V] (result)
```

### Độ phức tạp

| | GPU |
|---|---|
| Thời gian | O(V) vòng lặp × O(1) mỗi vòng (song song) |
| Không gian | O(V + E) trên VRAM |

---

## 📊 So sánh CPU vs GPU

| Tiêu chí | CPU | GPU (CUDA) |
|---|---|---|
| Xử lý cạnh | Tuần tự, từng cạnh một | Song song, tất cả cạnh cùng lúc |
| Số luồng | 1 | ~500.000 (= E) |
| Race condition | Không | Dùng `atomicMin` |
| Ghi log bước relax | (`RelaxationStep`) | |
| Phù hợp | Đồ thị nhỏ, debug | Đồ thị lớn, cần tốc độ |
| Độ phức tạp | O(V × E) | O(V) hiệu quả |

---

## Build & Chạy

### 1. Sinh đồ thị

```bash
python generate_graph.py
```

### 2. Compile

```bash
# Windows (MSVC + NVCC)
nvcc gpu_bellmanford.cu graph.cpp cpu_bellmanford.cpp main.cpp -o app.exe

# Linux (g++ + NVCC)
nvcc gpu_bellmanford.cu graph.cpp cpu_bellmanford.cpp main.cpp -o app -Xcompiler -O2
```

### 3. Chạy

```bash
./app.exe
```

Kết quả khoảng cách ngắn nhất được lưu vào thư mục `results/`, biểu đồ so sánh vào `visualization/`.

---

## Lưu ý

- Thuật toán Bellman-Ford **không phát hiện chu trình âm** trong phiên bản này (chỉ tìm đường ngắn nhất).
- Đồ thị sinh ra chỉ có **trọng số dương** (1–20), phù hợp cả Bellman-Ford lẫn Dijkstra.
- Kết quả CPU và GPU phải **khớp nhau** — dùng để kiểm tra tính đúng đắn của kernel CUDA.
