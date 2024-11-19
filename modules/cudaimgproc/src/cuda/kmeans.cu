#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaarithm.hpp>

using namespace cv::cuda;
using namespace cv;

#define CV_CUDA_CHECK(err)  __opencv_cuda_check((err), __FILE__, __LINE__)
inline void __opencv_cuda_check(cudaError_t err, const char *file, const int line)
{
    if (err != cudaSuccess)
    {
        CV_Error(cv::Error::GpuApiCallError, cudaGetErrorString(err));
    }
}

// Forward declarations
extern "C" {
    void printTimingStats();
}

namespace {

// Define distance constants for device code
const int KMEANS_DIST_L1 = 1;
const int KMEANS_DIST_L2 = 2;

// Define constants for shared memory and vectorized load optimization
#define TILE_DIM 32
#define MAX_DIMS 256  // Adjust based on your use case

// Utility function to compute L1/L2 distance between points
template<typename T>
__device__ float computeDistance(const T* a, const T* b, int dims, int distType) {
    float dist = 0.0f;
    for (int i = 0; i < dims; i++) {
        float diff = static_cast<float>(a[i]) - static_cast<float>(b[i]);
        if (distType == KMEANS_DIST_L1)
            dist += fabsf(diff);
        else // KMEANS_DIST_L2
            dist += diff * diff;
    }
    return (distType == KMEANS_DIST_L2) ? sqrtf(dist) : dist;
}

// Kernel for assigning points to nearest centers
template<typename T>
__global__ void assignCentersKernel(
    const PtrStepSz<T> data,
    const PtrStepSz<T> centers,
    PtrStepSz<int> labels,
    PtrStepSz<float> distances,
    const int distType
) {
    extern __shared__ float s_centers[];
    const int tid = threadIdx.x;
    const int point_idx = blockIdx.x * blockDim.x + tid;
    
    // Load centers into shared memory - handle each channel separately
    const int centers_per_thread = (centers.rows + blockDim.x - 1) / blockDim.x;
    
    // Load centers into shared memory
    #pragma unroll
    for (int k = 0; k < centers_per_thread; k++) {
        const int center_idx = k * blockDim.x + tid;
        if (center_idx < centers.rows) {
            // Load each channel separately to preserve exact values
            for (int c = 0; c < 3; c++) {
                s_centers[center_idx * 3 + c] = centers(center_idx, c);
            }
        }
    }
    __syncthreads();

    if (point_idx >= data.rows) return;

    float min_dist = FLT_MAX;
    int min_label = -1;
    
    // Load point data once
    float point_channels[3];
    for (int c = 0; c < 3; c++) {
        point_channels[c] = data(point_idx, c);
    }

    // Process all centers
    #pragma unroll
    for (int k = 0; k < centers.rows; k++) {
        float dist;
        if (distType == KMEANS_DIST_L1) {
            dist = fabsf(point_channels[0] - s_centers[k * 3 + 0]) + 
                   fabsf(point_channels[1] - s_centers[k * 3 + 1]) + 
                   fabsf(point_channels[2] - s_centers[k * 3 + 2]);
        } else {  // L2 distance
            float dx = point_channels[0] - s_centers[k * 3 + 0];
            float dy = point_channels[1] - s_centers[k * 3 + 1];
            float dz = point_channels[2] - s_centers[k * 3 + 2];
            dist = sqrtf(dx * dx + dy * dy + dz * dz);
        }

        if (dist < min_dist) {
            min_dist = dist;
            min_label = k;
        }
    }

    // Write results
    labels(point_idx, 0) = min_label;
    distances(point_idx, 0) = min_dist;
}

// Kernel for computing center means
template<typename T>
__global__ void updateCentersMeanKernel(
    PtrStepSz<T> centers,
    const int* center_counts,
    const int K
) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    const int count = center_counts[k];
    if (count > 0) {
        for (int dim = 0; dim < centers.cols; dim++) {
            centers(k, dim) /= count;
        }
    }
}

// Kernel for computing compactness (sum of distances)
__global__ void computeCompactnessKernel(
    const PtrStepSz<float> data,
    const PtrStepSz<int> labels,
    const PtrStepSz<float> centers,
    float* compactness,
    const int distType
) {
    extern __shared__ float shared_data[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    if (idx < data.rows) {
        int label = labels(idx, 0);
        if (label >= 0 && label < centers.rows) {
            float dist = computeDistance(data.ptr(idx), centers.ptr(label), data.cols, distType);
            sum = dist;
        }
    }
    shared_data[tid] = sum;
    __syncthreads();

    // Reduction to compute sum
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    // Write result to global memory
    if (tid == 0) {
        atomicAdd(compactness, shared_data[0]);
    }
}

// Kernel for computing center shift
__global__ void computeCenterShiftKernel(
    const PtrStepSz<float> new_centers,
    const PtrStepSz<float> old_centers,
    float* total_shift,
    const int distType
) {
    extern __shared__ float shared_data[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float shift = 0.0f;
    if (idx < new_centers.rows) {
        const float* new_center = new_centers.ptr(idx);
        const float* old_center = old_centers.ptr(idx);
        if (distType == KMEANS_DIST_L1) {
            for (int d = 0; d < new_centers.cols; d++) {
                shift += fabsf(new_center[d] - old_center[d]);
            }
        } else {
            for (int d = 0; d < new_centers.cols; d++) {
                float diff = new_center[d] - old_center[d];
                shift += diff * diff;
            }
        }
    }
    shared_data[tid] = shift;
    __syncthreads();

    // Reduction to compute sum
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    // Write result to global memory
    if (tid == 0) {
        atomicAdd(total_shift, shared_data[0]);
    }
}

// Kernel for gathering centers based on indices
__global__ void gatherCentersKernel(
    const PtrStepSz<float> data,
    const int* indices,
    PtrStepSz<float> centers
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= centers.rows) return;

    int idx = indices[k];
    const float* src = data.ptr(idx);
    float* dst = centers.ptr(k);

    for (int d = 0; d < data.cols; d++) {
        dst[d] = src[d];
    }
}

// Custom kernel for initializing closest_distances
__global__ void initializeClosestDistancesKernel(float* data, int N, float value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        data[idx] = value;
    }
}

// Custom kernel for element-wise minimum
__global__ void elementWiseMinKernel(const float* src1, const float* src2, float* dst, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        dst[idx] = fminf(src1[idx], src2[idx]);
    }
}

__global__ void selectNextCenterKernel(
    const float* distances,
    const float total_dist,
    const float random_value,
    int* selected_center,
    const int N
) {
    extern __shared__ float s_data[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;

    // Load distances into shared memory
    float local_sum = 0.0f;
    if (idx < N) {
        local_sum = distances[idx];
    }
    s_data[tid] = local_sum;
    __syncthreads();

    // Compute exclusive prefix sum in shared memory
    float running_sum = 0.0f;
    for (int i = 0; i < blockDim.x; i++) {
        float val = s_data[i];
        s_data[i] = running_sum;
        running_sum += val;
    }
    __syncthreads();

    // Each thread checks if the random value falls in its range
    if (idx < N) {
        float prev_sum = (bid > 0) ? distances[bid * blockDim.x - 1] : 0.0f;
        float my_start = prev_sum + s_data[tid];
        float my_end = prev_sum + s_data[tid] + local_sum;
        
        if (random_value >= my_start && random_value < my_end) {
            atomicExch(selected_center, idx);  // Use exchange instead of min
        }
    }
}

template<typename T>
__global__ void updateCentersSumCoalescedKernel(
    const PtrStepSz<T> data,        // Input data points [N][D]
    const PtrStepSz<int> labels,    // Labels for each point [N][1]
    PtrStepSz<T> centers,           // Center coordinates [K][D]
    int* center_counts,             // Points per center [K]
    const int K                     // Number of clusters
) {
    // Shared memory layout optimization:
    // Instead of doing atomic operations directly on global memory,
    // we first accumulate in shared memory which is much faster
    extern __shared__ float shared_mem[];
    // [K][BLOCK_DIMS] layout for better memory coalescing
    float* shared_centers = shared_mem;  
    // Place counts after centers data to maximize shared memory usage
    int* shared_counts = (int*)(shared_centers + K * blockDim.y);
    
    // Initialize shared memory to zeros
    // Each thread handles multiple centers if K > blockDim.x
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        // Each thread in y-dimension handles one dimension
        for (int d = threadIdx.y; d < blockDim.y; d += blockDim.y) {
            shared_centers[k * blockDim.y + d] = 0.0f;
        }
        // Only first row of threads initializes counts
        if (threadIdx.y == 0) {
            shared_counts[k] = 0;
        }
    }
    __syncthreads();  // Ensure initialization is complete

    // 2D Thread Block Optimization:
    // x-dimension: handles different points (blockDim.x = 256)
    // y-dimension: handles different dimensions (blockDim.y = 16)
    const int points_per_block = blockDim.x;
    const int point_idx = blockIdx.x * points_per_block + threadIdx.x;
    const int dim_idx = threadIdx.y;
    
    // Memory Coalescing Optimization:
    // Threads in the same warp access consecutive memory locations
    // This maximizes memory bandwidth utilization
    if (point_idx < data.rows && dim_idx < data.cols) {
        const int label = labels(point_idx, 0);
        if (label >= 0 && label < K) {
            // Atomic Add Optimization:
            // 1. Use shared memory atomics instead of global memory
            // 2. Memory access pattern is more cache-friendly
            // 3. Much less contention than global atomics
            atomicAdd(&shared_centers[label * blockDim.y + dim_idx], 
                     data(point_idx, dim_idx));
            
            // Count updates only needed once per point
            if (dim_idx == 0) {
                atomicAdd(&shared_counts[label], 1);
            }
        }
    }
    __syncthreads();  // Ensure all updates to shared memory are complete

    // Final Global Memory Update Optimization:
    // 1. Reduced number of global atomic operations
    // 2. Each block only needs one atomic update per center-dimension pair
    // 3. Better memory coalescing as threads update consecutive dimensions
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        if (dim_idx < data.cols) {
            atomicAdd(&centers(k, dim_idx), 
                     shared_centers[k * blockDim.y + dim_idx]);
        }
        // Only first row of threads updates counts
        if (threadIdx.y == 0) {
            atomicAdd(&center_counts[k], shared_counts[k]);
        }
    }
}

struct TimingStats {
    float kmeans_init_pp = 0;
    float kmeans_assign_labels = 0;
    float kmeans_update_centers = 0;
    float kmeans_center_shift = 0;
    float kmeans_compute_compactness = 0;
    int init_pp_calls = 0;
    int assign_labels_calls = 0;
    int update_centers_calls = 0;
    int center_shift_calls = 0;
    int compute_compactness_calls = 0;
};

static TimingStats timing_stats;

extern "C" void printTimingStats() {
    printf("\nKMeans CUDA Timing Summary:\n");
    printf("kmeans_init_pp_cuda:           %.2f ms (avg %.2f ms over %d calls)\n", 
           timing_stats.kmeans_init_pp,
           timing_stats.kmeans_init_pp / timing_stats.init_pp_calls,
           timing_stats.init_pp_calls);
    printf("kmeans_assign_labels_cuda:     %.2f ms (avg %.2f ms over %d calls)\n", 
           timing_stats.kmeans_assign_labels,
           timing_stats.kmeans_assign_labels / timing_stats.assign_labels_calls,
           timing_stats.assign_labels_calls);
    printf("kmeans_update_centers_cuda:    %.2f ms (avg %.2f ms over %d calls)\n", 
           timing_stats.kmeans_update_centers,
           timing_stats.kmeans_update_centers / timing_stats.update_centers_calls,
           timing_stats.update_centers_calls);
    printf("kmeans_center_shift_cuda:      %.2f ms (avg %.2f ms over %d calls)\n", 
           timing_stats.kmeans_center_shift,
           timing_stats.kmeans_center_shift / timing_stats.center_shift_calls,
           timing_stats.center_shift_calls);
    printf("kmeans_compute_compactness_cuda: %.2f ms (avg %.2f ms over %d calls)\n", 
           timing_stats.kmeans_compute_compactness,
           timing_stats.kmeans_compute_compactness / timing_stats.compute_compactness_calls,
           timing_stats.compute_compactness_calls);
    printf("Total CUDA kernel time: %.2f ms\n\n",
           timing_stats.kmeans_init_pp + 
           timing_stats.kmeans_assign_labels +
           timing_stats.kmeans_update_centers + 
           timing_stats.kmeans_center_shift +
           timing_stats.kmeans_compute_compactness);
}

__global__ void parallelReduceKernel(
    const float* input,
    float* output,
    int N
) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    // Load first round of data into shared memory
    float sum = (i < N) ? input[i] : 0;
    if (i + blockDim.x < N) 
        sum += input[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    // Do reduction in shared memory
    #pragma unroll
    for (unsigned int s = blockDim.x/2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Unroll last 6 iterations (warp)
    if (tid < 32) {
        volatile float* smem = sdata;
        if (blockDim.x >= 64) smem[tid] += smem[tid + 32];
        if (blockDim.x >= 32) smem[tid] += smem[tid + 16];
        if (blockDim.x >= 16) smem[tid] += smem[tid + 8];
        if (blockDim.x >= 8) smem[tid] += smem[tid + 4];
        if (blockDim.x >= 4) smem[tid] += smem[tid + 2];
        if (blockDim.x >= 2) smem[tid] += smem[tid + 1];
    }

    // Write result for this block to global mem
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

float customReduceSum(const GpuMat& input, Stream& stream) {
    const int N = input.rows * input.cols;
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + (threadsPerBlock * 2 - 1)) / (threadsPerBlock * 2);
    
    // Allocate temporary storage for block results
    GpuMat blockSums(1, blocksPerGrid, CV_32F);
    
    cudaStream_t cudaStream = StreamAccessor::getStream(stream);
    
    // First reduction pass
    parallelReduceKernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float), cudaStream>>>(
        input.ptr<float>(), blockSums.ptr<float>(), N
    );
    
    // If we have more than one block, do a second pass
    float total_sum;
    if (blocksPerGrid > 1) {
        GpuMat finalSum(1, 1, CV_32F);
        parallelReduceKernel<<<1, threadsPerBlock, threadsPerBlock * sizeof(float), cudaStream>>>(
            blockSums.ptr<float>(), finalSum.ptr<float>(), blocksPerGrid
        );
        cudaMemcpyAsync(&total_sum, finalSum.ptr<float>(), sizeof(float), 
                        cudaMemcpyDeviceToHost, cudaStream);
    } else {
        cudaMemcpyAsync(&total_sum, blockSums.ptr<float>(), sizeof(float), 
                        cudaMemcpyDeviceToHost, cudaStream);
    }
    
    stream.waitForCompletion();
    return total_sum;
}

} // namespace

// Implementation of CUDA interface functions

void initializeClosestDistances(GpuMat& mat, float value, Stream& stream)
{
    int N = mat.rows * mat.cols;
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Ensure gridSize does not exceed device limits
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int maxGridSizeX = deviceProp.maxGridSize[0];
    if (gridSize > maxGridSizeX)
    {
        gridSize = maxGridSizeX;
    }

    cudaStream_t cudaStream = StreamAccessor::getStream(stream);

    initializeClosestDistancesKernel<<<gridSize, blockSize, 0, cudaStream>>>(mat.ptr<float>(), N, value);

    CV_CUDA_CHECK(cudaGetLastError());
}

void elementWiseMin(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream)
{
    CV_Assert(src1.size() == src2.size());
    CV_Assert(src1.type() == CV_32F && src2.type() == CV_32F);

    int N = src1.rows * src1.cols;
    dst.create(src1.size(), src1.type());

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Ensure gridSize does not exceed device limits
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int maxGridSizeX = deviceProp.maxGridSize[0];
    if (gridSize > maxGridSizeX)
    {
        gridSize = maxGridSizeX;
    }

    cudaStream_t cudaStream = StreamAccessor::getStream(stream);

    elementWiseMinKernel<<<gridSize, blockSize, 0, cudaStream>>>(
        src1.ptr<float>(), src2.ptr<float>(), dst.ptr<float>(), N
    );

    CV_CUDA_CHECK(cudaGetLastError());
}

void kmeans_assign_labels_cuda(
    const GpuMat& data,
    const GpuMat& centers,
    GpuMat& labels,
    GpuMat& distances,
    int distType,
    int blockSize,
    Stream& stream
) {
    // Create and record start event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, StreamAccessor::getStream(stream));
    
    CV_Assert(data.type() == CV_32F);
    CV_Assert(centers.type() == CV_32F);
    int cudaDistType = (distType == DistanceTypes::DIST_L1) ? KMEANS_DIST_L1 : KMEANS_DIST_L2;
    const int rows = data.rows;
    
    if (blockSize <= 0)
        blockSize = 512;

    // Device limits check
    int maxThreadsPerBlock;
    cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);
    blockSize = std::min(blockSize, maxThreadsPerBlock);
    
    // Calculate shared memory size for centers
    size_t shared_mem_size = centers.rows * sizeof(float3);
    int maxSharedMemPerBlock;
    cudaDeviceGetAttribute(&maxSharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    if (shared_mem_size > (size_t)maxSharedMemPerBlock) {
        blockSize = 256;
        shared_mem_size = 0;
    }
    
    dim3 block(blockSize);
    dim3 grid((rows + block.x - 1) / block.x);
    cudaStream_t cudaStream = StreamAccessor::getStream(stream);
    
    // Launch kernel with shared memory
    assignCentersKernel<float><<<grid, block, shared_mem_size, cudaStream>>>(
        data, centers, labels, distances, cudaDistType
    );
    cudaStreamSynchronize(cudaStream);

    CV_CUDA_CHECK(cudaGetLastError());
    
    // Record stop event and calculate time
    cudaEventRecord(stop, StreamAccessor::getStream(stream));
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    timing_stats.kmeans_assign_labels += milliseconds;
    timing_stats.assign_labels_calls++;

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void kmeans_update_centers_cuda(const GpuMat& data, const GpuMat& labels,
                               GpuMat& centers, GpuMat& center_counts,
                               int K, int blockSize, Stream& stream)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, StreamAccessor::getStream(stream));

    // Clear centers and counts
    centers.setTo(Scalar::all(0), stream);
    center_counts.setTo(Scalar::all(0), stream);

    const int BLOCK_DIMS = 16; // Number of dimensions to process per block
    dim3 block(256, BLOCK_DIMS); // 256 points × 16 dimensions per block
    
    // Calculate grid size for points
    const int grid_x = (data.rows + block.x - 1) / block.x;
    dim3 grid(grid_x);

    // Calculate shared memory size
    size_t shared_mem_size = (K * BLOCK_DIMS * sizeof(float)) + (K * sizeof(int));
    
    cudaStream_t cudaStream = StreamAccessor::getStream(stream);

    // Process centers in chunks of BLOCK_DIMS dimensions
    for (int dim_offset = 0; dim_offset < data.cols; dim_offset += BLOCK_DIMS) {
        const int dims_remaining = std::min(BLOCK_DIMS, data.cols - dim_offset);
        block.y = dims_remaining;
        
        updateCentersSumCoalescedKernel<float><<<grid, block, shared_mem_size, cudaStream>>>(
            data, labels, centers, center_counts.ptr<int>(), K
        );
        CV_CUDA_CHECK(cudaGetLastError());
    }

    // Update centers by dividing by counts
    const dim3 blockMean(256);
    const dim3 gridMean((K + blockMean.x - 1) / blockMean.x);

    updateCentersMeanKernel<float><<<gridMean, blockMean, 0, cudaStream>>>(
        centers, center_counts.ptr<int>(), K
    );

    CV_CUDA_CHECK(cudaGetLastError());

    // Timing code...
    cudaEventRecord(stop, cudaStream);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    timing_stats.kmeans_update_centers += milliseconds;
    timing_stats.update_centers_calls++;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void kmeans_random_center_init_cuda(const GpuMat& data, int K, GpuMat& centers, RNG& rng, Stream& stream)
{
    const int N = data.rows;

    // Generate K random indices on the CPU
    std::vector<int> indices(K);
    for (int i = 0; i < K; i++)
        indices[i] = rng.uniform(0, N);

    // Upload indices to GPU
    GpuMat indices_gpu(K, 1, CV_32S);
    indices_gpu.upload(indices, stream);

    // Use gather operation to select the centers
    int blockSize = 256;

    // Ensure blockSize does not exceed device limit
    int maxThreadsPerBlock;
    cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);
    blockSize = std::min(blockSize, maxThreadsPerBlock);

    const dim3 block(blockSize);
    const dim3 grid((K + block.x - 1) / block.x);

    cudaStream_t cudaStream = StreamAccessor::getStream(stream);

    gatherCentersKernel<<<grid, block, 0, cudaStream>>>(
        data, indices_gpu.ptr<int>(), centers
    );

    CV_CUDA_CHECK(cudaGetLastError());
}

void kmeans_init_pp_cuda(const GpuMat& data, int K, GpuMat& centers,
                         GpuMat& distances, RNG& rng, int blockSize, Stream& stream)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, StreamAccessor::getStream(stream));

    const int N = data.rows;

    // Initialize first center randomly
    int center_idx = rng.uniform(0, N);
    data.row(center_idx).copyTo(centers.row(0), stream);

    if (blockSize <= 0)
        blockSize = 256;

    // Ensure blockSize does not exceed device limit
    int maxThreadsPerBlock;
    cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);
    blockSize = std::min(blockSize, maxThreadsPerBlock);

    // Allocate GPU memory
    GpuMat closest_distances(N, 1, CV_32F);
    GpuMat temp_labels(N, 1, CV_32S);
    GpuMat d_total_dist(1, 1, CV_32F);
    GpuMat d_selected_center(1, 1, CV_32S);
    distances.create(N, 1, CV_32F);

    // Initialize distances
    initializeClosestDistances(closest_distances, FLT_MAX, stream);

    // Ensure shared memory size doesn't exceed device limit
    size_t sharedMemSize = blockSize * sizeof(float);
    int maxSharedMemPerBlock;
    cudaDeviceGetAttribute(&maxSharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    if (sharedMemSize > (size_t)maxSharedMemPerBlock) {
        sharedMemSize = maxSharedMemPerBlock;
        blockSize = sharedMemSize / sizeof(float);
    }

    const dim3 block(blockSize);
    const dim3 grid((N + block.x - 1) / block.x);
    cudaStream_t cudaStream = StreamAccessor::getStream(stream);

    for (int k = 1; k < K; k++) {
        // Assign labels to closest centers
        kmeans_assign_labels_cuda(data, centers.rowRange(0, k),
                                 temp_labels, distances,
                                 DistanceTypes::DIST_L2, blockSize, stream);

        // Update closest distances
        elementWiseMin(closest_distances, distances, closest_distances, stream);

        // Compute total distance on GPU
        float total_dist = customReduceSum(closest_distances, stream);

        // Generate random value
        float r = static_cast<float>(rng.uniform(0.0, static_cast<double>(total_dist)));

        // Select new center on GPU
        d_selected_center.setTo(Scalar(N), stream);

        selectNextCenterKernel<<<grid, block, sharedMemSize, cudaStream>>>(
            closest_distances.ptr<float>(), total_dist, r,
            d_selected_center.ptr<int>(), N
        );
        CV_CUDA_CHECK(cudaGetLastError());

        // Get selected center index
        int new_center_idx;
        stream.waitForCompletion();
        cudaMemcpyAsync(&new_center_idx, d_selected_center.ptr<int>(), sizeof(int),
                       cudaMemcpyDeviceToHost, cudaStream);
        stream.waitForCompletion();

        if (new_center_idx >= N) {
            new_center_idx = N - 1; // Fallback if no valid center found
        }

        // Copy the new center
        data.row(new_center_idx).copyTo(centers.row(k), stream);
    }

    cudaEventRecord(stop, StreamAccessor::getStream(stream));
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    timing_stats.kmeans_init_pp += milliseconds;
    timing_stats.init_pp_calls++;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

double kmeans_center_shift_cuda(const GpuMat& new_centers, const GpuMat& old_centers,
                                int distType, int blockSize, Stream& stream)
{
    // Create and record start event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, StreamAccessor::getStream(stream));

    const int K = new_centers.rows;

    GpuMat d_total_shift(1, 1, CV_32F);
    d_total_shift.setTo(Scalar::all(0), stream);

    if (blockSize <= 0)
        blockSize = 256; // Default block size

    // Ensure blockSize does not exceed device limit
    int maxThreadsPerBlock;
    cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);
    blockSize = std::min(blockSize, maxThreadsPerBlock);

    size_t sharedMemSize = blockSize * sizeof(float);

    // Ensure sharedMemSize does not exceed device limit
    int maxSharedMemPerBlock;
    cudaDeviceGetAttribute(&maxSharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    if (sharedMemSize > (size_t)maxSharedMemPerBlock)
    {
        sharedMemSize = maxSharedMemPerBlock;
        blockSize = sharedMemSize / sizeof(float);
    }

    const dim3 block(blockSize);
    const dim3 grid((K + block.x - 1) / block.x);

    cudaStream_t cudaStream = StreamAccessor::getStream(stream);

    computeCenterShiftKernel<<<grid, block, sharedMemSize, cudaStream>>>(
        new_centers, old_centers, d_total_shift.ptr<float>(), distType
    );

    CV_CUDA_CHECK(cudaGetLastError());

    // Copy total shift back to host
    float total_shift = 0.0f;
    cudaMemcpyAsync(&total_shift, d_total_shift.ptr<float>(), sizeof(float), cudaMemcpyDeviceToHost, cudaStream);
    stream.waitForCompletion();

    // Record stop event and calculate time
    cudaEventRecord(stop, cudaStream);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    timing_stats.kmeans_center_shift += milliseconds;
    timing_stats.center_shift_calls++;

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return static_cast<double>(total_shift);
}

double kmeans_compute_compactness_cuda(const GpuMat& data, const GpuMat& labels,
                                       const GpuMat& centers, GpuMat& distances,
                                       int distType, int blockSize, Stream& stream)
{
    // Create and record start event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, StreamAccessor::getStream(stream));

    const int N = data.rows;

    if (blockSize <= 0)
        blockSize = 256; // Default block size

    // Ensure blockSize does not exceed device limit
    int maxThreadsPerBlock;
    cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);
    blockSize = std::min(blockSize, maxThreadsPerBlock);

    size_t sharedMemSize = blockSize * sizeof(float);

    // Ensure sharedMemSize does not exceed device limit
    int maxSharedMemPerBlock;
    cudaDeviceGetAttribute(&maxSharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    if (sharedMemSize > (size_t)maxSharedMemPerBlock)
    {
        sharedMemSize = maxSharedMemPerBlock;
        blockSize = sharedMemSize / sizeof(float);
    }

    const dim3 block(blockSize);
    const dim3 grid((N + block.x - 1) / block.x);

    cudaStream_t cudaStream = StreamAccessor::getStream(stream);

    // Allocate memory for compactness result on device
    GpuMat d_compactness(1, 1, CV_32F);
    d_compactness.setTo(Scalar::all(0), stream);

    computeCompactnessKernel<<<grid, block, sharedMemSize, cudaStream>>>(
        data, labels, centers, d_compactness.ptr<float>(), distType
    );

    CV_CUDA_CHECK(cudaGetLastError());

    // Copy compactness result back to host
    float compactness = 0.0f;
    cudaMemcpyAsync(&compactness, d_compactness.ptr<float>(), sizeof(float), cudaMemcpyDeviceToHost, cudaStream);
    stream.waitForCompletion();

    // Record stop event and calculate time
    cudaEventRecord(stop, cudaStream);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    timing_stats.kmeans_compute_compactness += milliseconds;
    timing_stats.compute_compactness_calls++;

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return static_cast<double>(compactness);
}
