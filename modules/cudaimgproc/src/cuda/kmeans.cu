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

namespace {

// Define distance constants for device code
const int KMEANS_DIST_L1 = 1;
const int KMEANS_DIST_L2 = 2;

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
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= data.rows) return;

    float minDist = FLT_MAX;
    int minLabel = 0;

    // Find nearest center
    for (int k = 0; k < centers.rows; k++) {
        float dist = computeDistance(data.ptr(idx), centers.ptr(k), data.cols, distType);
        if (dist < minDist) {
            minDist = dist;
            minLabel = k;
        }
    }

    labels(idx, 0) = minLabel;
    distances(idx, 0) = minDist;
}

// Kernel for updating centers
template<typename T>
__global__ void updateCentersSumKernel(
    const PtrStepSz<T> data,
    const PtrStepSz<int> labels,
    PtrStepSz<T> centers,
    int* center_counts,
    const int K
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= data.rows) return;

    const int label = labels(idx, 0);
    if (label >= 0 && label < K) {
        for (int dim = 0; dim < data.cols; dim++) {
            atomicAdd(&centers(label, dim), data(idx, dim));
        }
        atomicAdd(&center_counts[label], 1);
    }
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

void kmeans_assign_labels_cuda(const GpuMat& data, const GpuMat& centers,
                               GpuMat& labels, GpuMat& distances,
                               int distType, int blockSize, Stream& stream)
{
    CV_Assert(data.type() == CV_32F);
    CV_Assert(centers.type() == CV_32F);

    int cudaDistType = (distType == DistanceTypes::DIST_L1) ? KMEANS_DIST_L1 : KMEANS_DIST_L2;

    const int rows = data.rows;

    // Ensure labels and distances are allocated
    if (labels.empty() || labels.rows != rows || labels.cols != 1)
        labels.create(rows, 1, CV_32S);
    if (distances.empty() || distances.rows != rows || distances.cols != 1)
        distances.create(rows, 1, CV_32F);

    if (blockSize <= 0)
        blockSize = 256; // Default block size

    // Ensure blockSize does not exceed device limit
    int maxThreadsPerBlock;
    cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);
    blockSize = std::min(blockSize, maxThreadsPerBlock);

    const dim3 block(blockSize);
    const dim3 grid((rows + block.x - 1) / block.x);

    cudaStream_t cudaStream = StreamAccessor::getStream(stream);

    assignCentersKernel<float><<<grid, block, 0, cudaStream>>>(
        data, centers, labels, distances, cudaDistType
    );

    CV_CUDA_CHECK(cudaGetLastError());
}

void kmeans_update_centers_cuda(const GpuMat& data, const GpuMat& labels,
                                GpuMat& centers, GpuMat& center_counts,
                                int K, int blockSize, Stream& stream)
{
    const int rows = data.rows;

    centers.setTo(Scalar::all(0), stream);
    center_counts.setTo(Scalar::all(0), stream);

    if (blockSize <= 0)
        blockSize = 256; // Default block size

    // Ensure blockSize does not exceed device limit
    int maxThreadsPerBlock;
    cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);
    blockSize = std::min(blockSize, maxThreadsPerBlock);

    const dim3 block(blockSize);
    const dim3 grid((rows + block.x - 1) / block.x);

    cudaStream_t cudaStream = StreamAccessor::getStream(stream);

    // Allocate memory for center_counts on device
    int* d_center_counts = (int*)center_counts.data;

    updateCentersSumKernel<float><<<grid, block, 0, cudaStream>>>(
        data, labels, centers, d_center_counts, K
    );

    CV_CUDA_CHECK(cudaGetLastError());

    // Update centers by dividing by counts
    const dim3 blockMean(blockSize);
    const dim3 gridMean((K + blockMean.x - 1) / blockMean.x);

    updateCentersMeanKernel<float><<<gridMean, blockMean, 0, cudaStream>>>(
        centers, d_center_counts, K
    );

    CV_CUDA_CHECK(cudaGetLastError());
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
    const int N = data.rows;

    // Initialize first center randomly
    int center_idx = rng.uniform(0, N);
    data.row(center_idx).copyTo(centers.row(0), stream);

    if (blockSize <= 0)
        blockSize = 256; // Default block size

    // Ensure blockSize does not exceed device limit
    int maxThreadsPerBlock;
    cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);
    blockSize = std::min(blockSize, maxThreadsPerBlock);

    GpuMat closest_distances(N, 1, CV_32F);

    // Use custom function to initialize closest_distances
    initializeClosestDistances(closest_distances, FLT_MAX, stream);

    // Allocate temp_labels and distances
    GpuMat temp_labels(N, 1, CV_32S);
    distances.create(N, 1, CV_32F);

    for (int k = 1; k < K; k++) {
        // Assign labels to the closest center
        kmeans_assign_labels_cuda(data, centers.rowRange(0, k),
                                  temp_labels, distances,
                                  DistanceTypes::DIST_L2, blockSize, stream);

        // Update closest distances using custom element-wise min
        elementWiseMin(closest_distances, distances, closest_distances, stream);

        // Synchronize the stream before proceeding
        stream.waitForCompletion();

        // Copy closest distances to host
        Mat h_closest_distances;
        closest_distances.download(h_closest_distances, stream);
        stream.waitForCompletion();

        // Compute total distance on host
        double total_dist = cv::sum(h_closest_distances)[0];

        // Generate a random value in [0, total_dist)
        double r = rng.uniform(0.0, total_dist);

        // Find the index where cumulative distance exceeds r
        double cumulative = 0.0;
        int new_center_idx = -1;
        for (int i = 0; i < N; ++i) {
            cumulative += h_closest_distances.at<float>(i, 0);
            if (cumulative >= r) {
                new_center_idx = i;
                break;
            }
        }

        if (new_center_idx == -1) {
            new_center_idx = N - 1; // Fallback
        }

        // Copy the new center
        data.row(new_center_idx).copyTo(centers.row(k), stream);

        // Re-initialize distances to FLT_MAX for the next iteration
        initializeClosestDistances(closest_distances, FLT_MAX, stream);
    }
}


double kmeans_center_shift_cuda(const GpuMat& new_centers, const GpuMat& old_centers,
                                int distType, int blockSize, Stream& stream)
{
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

    return static_cast<double>(total_shift);
}

double kmeans_compute_compactness_cuda(const GpuMat& data, const GpuMat& labels,
                                       const GpuMat& centers, GpuMat& distances,
                                       int distType, int blockSize, Stream& stream)
{
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

    return static_cast<double>(compactness);
}
