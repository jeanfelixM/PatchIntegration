__global__ void filterPatch(float* p2, float* patchtarget, int rows, int boundcols, int boundrows){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;

    int targetX = static_cast<int>(p2[2 * i]); //row-major
    int targetY = static_cast<int>(p2[2 * i + 1]);

    if (targetX >= 0 && targetX < boundcols && targetY >= 0 && targetY < boundrows) {
        patchtarget[2 * i] = static_cast<float>(targetX);
        patchtarget[2 * i + 1] = static_cast<float>(targetY);
    } else {
        patchtarget[2 * i] = -1; 
        patchtarget[2 * i + 1] = -1;
    }
}

__global__ void multiply(float* p1, float* p2, float* p3, int rows){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;

    p3[2 * i] = p1[2 * i] * p2[2 * i];
    p3[2 * i + 1] = p1[2 * i + 1] * p2[2 * i + 1];
}

__global__ void divide(float* p1, float* p2, float* p3, int rows){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;

    p3[2 * i] = p1[2 * i] / p2[2 * i];
    p3[2 * i + 1] = p1[2 * i + 1] / p2[2 * i + 1];
}

__global__ void vconcat(float* A, int A_rows, float* B, int B_rows, float* C, int C_rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < C_rows && col < cols) {
        if (row < A_rows) {
            // Copier depuis la matrice A
            C[row * cols + col] = A[row * cols + col];
        } else {
            // Copier depuis la matrice B
            C[row * cols + col] = B[(row - A_rows) * cols + col];
        }
    }
}

__global__ void hconcat(float* A, int A_cols, float* B, int B_cols, float* C, int C_cols, int rows) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < C_cols) {
        if (col < A_cols) {
            // Copier depuis la matrice A
            C[row * C_cols + col] = A[row * A_cols + col];
        } else {
            // Copier depuis la matrice B
            C[row * C_cols + col] = B[row * B_cols + (col - A_cols)];
        }
    }
}

__global__ float zncc(float* patch1, float* patch2, int rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return 0;

    float mean1 = 0;
    float mean2 = 0;
    float std1 = 0;
    float std2 = 0;
    float cov = 0;

    for (int j = 0; j < rows; j++) {
        mean1 += patch1[j];
        mean2 += patch2[j];
    }
    mean1 /= rows;
    mean2 /= rows;

    for (int j = 0; j < rows; j++) {
        std1 += (patch1[j] - mean1) * (patch1[j] - mean1);
        std2 += (patch2[j] - mean2) * (patch2[j] - mean2);
        cov += (patch1[j] - mean1) * (patch2[j] - mean2);
    }
    std1 = sqrt(std1);
    std2 = sqrt(std2);

    return cov / (std1 * std2);
}

__global__ void mean(float* patch, int rows, float* mean) {
    __shared__ float sum[rows];
    reduce_sum(sum, patch, rows);

    mean[2*blockIdx.x] = sum[2*blockIdx.x] / rows;
    mean[2*blockIdx.x + 1] = sum[2*blockIdx.x + 1] / rows;
}


__device__ void reduce_sum(float* g_odata, float* g_idata, unsigned int len) {
	extern __shared__ float sdata[];

	// each thread loads one element from global to shared mem
	// Do the first stage of the reduction on the global-to-shared load step
	// This reduces the previous inefficiency of having half of the threads being
	//  inactive on the first for-loop iteration below (previous first step of reduction)
	// Previously, only less than or equal to 512 out of 1024 threads in a block are active.
	// Now, all 512 threads in a block are active from the start
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	sdata[tid] = 0.0;

	if (i < len)
	{
		sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
	}

	__syncthreads();

	// do reduction in shared mem
	// this loop now starts with s = 512 / 2 = 256
	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	if (tid < 32)
	{
		sdata[tid] += sdata[tid + 32];
		sdata[tid] += sdata[tid + 16];
		sdata[tid] += sdata[tid + 8];
		sdata[tid] += sdata[tid + 4];
		sdata[tid] += sdata[tid + 2];
		sdata[tid] += sdata[tid + 1];
	}

	// write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
}

__device__ void reduce_mul(unsigned int* g_odata, unsigned int* g_idata, unsigned int len) {
	extern __shared__ unsigned int sdata[];

	// each thread loads one element from global to shared mem
	// Do the first stage of the reduction on the global-to-shared load step
	// This reduces the previous inefficiency of having half of the threads being
	//  inactive on the first for-loop iteration below (previous first step of reduction)
	// Previously, only less than or equal to 512 out of 1024 threads in a block are active.
	// Now, all 512 threads in a block are active from the start
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	sdata[tid] = 0;

	if (i < len)
	{
		sdata[tid] = g_idata[i] * g_idata[i + blockDim.x];
	}

	__syncthreads();

	// do reduction in shared mem
	// this loop now starts with s = 512 / 2 = 256
	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s) {
			sdata[tid] *= sdata[tid + s];
		}
		__syncthreads();
	}

	if (tid < 32)
	{
		sdata[tid] *= sdata[tid + 32];
		sdata[tid] *= sdata[tid + 16];
		sdata[tid] *= sdata[tid + 8];
		sdata[tid] *= sdata[tid + 4];
		sdata[tid] *= sdata[tid + 2];
		sdata[tid] *= sdata[tid + 1];
	}

	// write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
}