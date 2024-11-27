#include "precomp.hpp"

using namespace cv;
using namespace cv::cuda;

#if !defined (HAVE_CUDA) || defined (CUDA_DISABLER)

double cv::cuda::KMeans::cluster(InputArray, int, InputOutputArray,
                                 TermCriteria, OutputArray, Stream&)
{
    throw_no_cuda();
    return 0.0;
}

Ptr<cv::cuda::KMeans> cv::cuda::createKMeans(int, int, int, int)
{
    throw_no_cuda();
    return Ptr<cv::cuda::KMeans>();
}

#else /* !defined (HAVE_CUDA) */

// Forward declarations of CUDA functions
void kmeans_cuda(const GpuMat& data, int K, GpuMat& labels, GpuMat& centers,
                 GpuMat& distances, int distType, int blockSize, Stream& stream);
void kmeans_init_pp_cuda(const GpuMat& data, int K, GpuMat& centers,
                         GpuMat& distances, RNG& rng, int blockSize, Stream& stream);
void kmeans_assign_labels_cuda(const GpuMat& data, const GpuMat& centers,
                               GpuMat& labels, GpuMat& distances,
                               int distType, int blockSize, Stream& stream);
void kmeans_update_centers_cuda(const GpuMat& data, const GpuMat& labels,
                                GpuMat& centers, GpuMat& center_counts,
                                int K, int blockSize, Stream& stream);
double kmeans_compute_compactness_cuda(const GpuMat& data, const GpuMat& labels,
                                       const GpuMat& centers, GpuMat& distances,
                                       int distType, int blockSize, Stream& stream);
double kmeans_center_shift_cuda(const GpuMat& new_centers, const GpuMat& old_centers,
                                int distType, int blockSize, Stream& stream);
void kmeans_random_center_init_cuda(const GpuMat& data, int K, GpuMat& centers, RNG& rng, Stream& stream);

class KMeans_Impl : public cv::cuda::KMeans
{
public:
    KMeans_Impl(int initMethod, int attempts, int distType, int blockSize) :
        initMethod_(initMethod),
        attempts_(attempts),
        distType_(distType),
        blockSize_(blockSize)
    {
    }

    double cluster(InputArray _data, int K,
                   InputOutputArray _bestLabels,
                   TermCriteria criteria,
                   OutputArray _centers,
                   Stream& stream) CV_OVERRIDE
    {
        // Verify input parameters
        CV_Assert(K > 0);
        CV_Assert(attempts_ >= 1);
        CV_Assert(_data.type() == CV_32F);

        GpuMat data = _data.getGpuMat();
        const int N = data.rows;
        const int dims = data.cols;
        CV_Assert(N >= K);

        // Create output matrices if needed
        _bestLabels.create(N, 1, CV_32S);
        GpuMat labels = _bestLabels.getGpuMat();

        GpuMat centers;
        if (_centers.needed())
        {
            _centers.create(K, dims, CV_32F);
            centers = _centers.getGpuMat();
        }

        // Allocate temporary buffers
        ensureBufferSize(N, K, dims);

        double best_compactness = DBL_MAX;

        for (int a = 0; a < attempts_; a++)
        {
            GpuMat cur_labels(N, 1, CV_32S);
            GpuMat cur_centers(K, dims, CV_32F);

            // Initialize centers
            if (a == 0 && (initMethod_ & KMEANS_USE_INITIAL_LABELS))
            {
                labels.copyTo(cur_labels);

                // Compute initial centers from labels
                center_counts_.setTo(0, stream);
                kmeans_update_centers_cuda(data, cur_labels, cur_centers,
                                           center_counts_, K, blockSize_, stream);
            }
            else
            {
                // Initialize centers using specified method
                if (initMethod_ & KMEANS_PP_CENTERS)
                {
                    RNG& rng = theRNG();
                    kmeans_init_pp_cuda(data, K, cur_centers, distances_,
                                        rng, blockSize_, stream);
                }
                else
                {
                    // Random initialization entirely on the GPU
                    RNG& rng = theRNG();
                    kmeans_random_center_init_cuda(data, K, cur_centers, rng, stream);
                }
            }

            // K-means iteration
            GpuMat old_centers(K, dims, CV_32F);
            double compactness = 0;

            for (int iter = 0;; iter++)
            {
                cur_centers.copyTo(old_centers, stream);

                // Assign labels
                kmeans_assign_labels_cuda(data, cur_centers, cur_labels,
                                          distances_, distType_, blockSize_, stream);

                // Update centers
                center_counts_.setTo(0, stream);
                kmeans_update_centers_cuda(data, cur_labels, cur_centers,
                                           center_counts_, K, blockSize_, stream);

                // Check convergence
                if (iter > 0)
                {
                    // Compute center shift using CUDA
                    double center_shift = kmeans_center_shift_cuda(cur_centers, old_centers,
                                                                   distType_, blockSize_, stream);
                    if (iter >= criteria.maxCount || center_shift <= criteria.epsilon)
                    {
                        // Compute final compactness
                        compactness = kmeans_compute_compactness_cuda(
                            data, cur_labels, cur_centers, distances_,
                            distType_, blockSize_, stream);
                        break;
                    }
                }
            }

            // Update best result if needed
            if (compactness < best_compactness)
            {
                best_compactness = compactness;
                cur_labels.copyTo(labels, stream);
                if (!centers.empty())
                    cur_centers.copyTo(centers, stream);
            }
        }

        // Ensure all CUDA operations are completed
        stream.waitForCompletion();

        return best_compactness;
    }

    // Getter/Setter implementations
    int getInitMethod() const CV_OVERRIDE { return initMethod_; }
    void setInitMethod(int method) CV_OVERRIDE
    {
        CV_Assert(method == KMEANS_RANDOM_CENTERS ||
                  method == KMEANS_PP_CENTERS ||
                  method == KMEANS_USE_INITIAL_LABELS ||
                  method == (KMEANS_PP_CENTERS | KMEANS_USE_INITIAL_LABELS) ||
                  method == (KMEANS_RANDOM_CENTERS | KMEANS_USE_INITIAL_LABELS));
        initMethod_ = method;
    }

    int getAttempts() const CV_OVERRIDE { return attempts_; }
    void setAttempts(int attempts) CV_OVERRIDE
    {
        CV_Assert(attempts > 0);
        attempts_ = attempts;
    }

    int getDistanceType() const CV_OVERRIDE { return distType_; }
    void setDistanceType(int distType) CV_OVERRIDE
    {
        CV_Assert(distType == DIST_L1 || distType == DIST_L2);
        distType_ = distType;
    }

    int getBlockSize() const CV_OVERRIDE { return blockSize_; }
    void setBlockSize(int blockSize) CV_OVERRIDE
    {
        CV_Assert(blockSize >= 0);
        blockSize_ = blockSize;
    }

private:
    void ensureBufferSize(int N, int K, int dims)
    {
        distances_.create(N, 1, CV_32F);
        center_counts_.create(K, 1, CV_32S);
    }

    int initMethod_;
    int attempts_;
    int distType_;
    int blockSize_;

    // Temporary buffers
    GpuMat distances_;      // Distance from each point to its center
    GpuMat center_counts_;  // Number of points in each cluster
};

Ptr<KMeans> cv::cuda::createKMeans(int initMethod, int attempts,
                                   int distType, int blockSize)
{
    return makePtr<KMeans_Impl>(initMethod, attempts, distType, blockSize);
}

#endif /* !defined (HAVE_CUDA) */
