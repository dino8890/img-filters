#include <cstdio>

__global__ void apply_filter(unsigned char *red_channel, unsigned char *green_channel, unsigned char *blue_channel,
                             unsigned int width, unsigned int height) {

    unsigned int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    unsigned int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.x + threadIdx.y * blockDim.x);

    if(threadId < width * height) {
        // according to the NTSC/ATSC standard
        // intensity = (0.2126 * red_value + 0.7152 * green_value + 0.0722 * blue_value)
        unsigned char intensity = static_cast<unsigned char>(
            red_channel[threadId] * 0.2126 + green_channel[threadId] * 0.7152 + blue_channel[threadId] * 0.0722
        );

        red_channel[threadId] = intensity;
        green_channel[threadId] = intensity;
        blue_channel[threadId] = intensity;
    }
}
