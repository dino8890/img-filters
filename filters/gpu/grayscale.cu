__global__ void applyFilter(unsigned char *redChannel,
                            unsigned char *greenChannel,
                            unsigned char *blueChannel,
                            const unsigned int width, const unsigned int height) {
    const unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
    const unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;

    if(row < height && col < width) {
        // according to the NTSC/ATSC standard
        // intensity = (0.2126 * red_value + 0.7152 * green_value + 0.0722 * blue_value)
        const unsigned int index = col + row * width;
        const unsigned char intensity = static_cast<unsigned char>(
            redChannel[index] * 0.2126 + greenChannel[index] * 0.7152 + blueChannel[index] * 0.0722
        );

        redChannel[index] = intensity;
        greenChannel[index] = intensity;
        blueChannel[index] = intensity;
    }
}
