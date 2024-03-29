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
            C[row * cols + col] = A[row * cols + col];
        } else {
            C[row * cols + col] = B[(row - A_rows) * cols + col];
        }
    }
}

__global__ void hconcat(float* A, int A_cols, float* B, int B_cols, float* C, int C_cols, int rows) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < C_cols) {
        if (col < A_cols) {
            C[row * C_cols + col] = A[row * A_cols + col];
        } else {
            C[row * C_cols + col] = B[row * B_cols + (col - A_cols)];
        }
    }
}

__global__ float variance(float* patch1, float* patch2, float* std1, float* std2, float* cov, float mean1, float mean2,int rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return 0;

    std1[i] = (patch1[i] - mean1) * (patch1[i] - mean1);
    std2[i] = (patch2[i] - mean2) * (patch2[i] - mean2);
    cov[i] = (patch1[i] - mean1) * (patch2[i] - mean2);

}

__global__ void copatch_to_impatch(float* im, float* patch, int rows,float* impatch){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return 0;

    int x = __float2int_rd(patch[2*i]);
    int y = __float2int_rd(patch[2*i+1]);
    impatch[i] = im[x*m + y];
}


//si j'ai bien tout compris c'est quasi le mieux possible, source : https://github.com/mark-poscablo/gpu-sum-reduction
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

