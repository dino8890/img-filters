__global__ void apply_filter(const unsigned char *input_channel, unsigned char *output_channel,
                             const unsigned int width, const unsigned int height,
                             const float *gaussian_kernel, const unsigned int filter_width) {
    const unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
    const unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
    if(row < height && col < width) {
        const int filter_half = filter_width / 2;
        float blur = 0.0;
        for(int i = -filter_half; i <= filter_half; i++) {
            for(int j = -filter_half; j <= filter_half; j++) {
                const unsigned int y = max(0, min(height - 1, row + i));
                const unsigned int x = max(0, min(width - 1, col + j));

                const float w = gaussian_kernel[(j + filter_half) + (i + filter_half) * filter_width];
                blur += w * input_channel[x + y * width];
            }
        }
        output_channel[col + row * width] = static_cast<unsigned char>(blur);
    }
}
