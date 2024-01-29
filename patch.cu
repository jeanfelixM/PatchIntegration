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