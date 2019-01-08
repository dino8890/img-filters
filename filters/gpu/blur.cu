__global__ void apply_filter(unsigned char *input_channel, unsigned char *output_channel, unsigned char *buffer,
                             unsigned int width, unsigned int height,
                             float *gaussian_kernel, float standard_deviation, unsigned int filter_width) {

    unsigned int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    unsigned int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.x + threadIdx.y * blockDim.x);
    if(threadId < width * height) {
        unsigned int row = threadIdx.y + blockIdx.y * gridDim.y;
        unsigned int col = threadIdx.x + blockIdx.x * gridDim.x;

        buffer[col + row * height] = input_channel[threadId];

        __syncthreads();

        float blur = 0.0;
        int filter_half = filter_width / 2;
        for(int i = -filter_half; i <= filter_half; i++) {
            for(int j = -filter_half; j <= filter_half; j++) {
                unsigned int x = max(0, min(height - 1, col + j));
                unsigned int y = max(0, min(width - 1, row + i));

                blur += buffer[x + y * height] * gaussian_kernel[(j + filter_half) + (i + filter_half) * filter_width];
            }
        }

        output_channel[threadId] = static_cast<unsigned char>(blur);
    }
}
