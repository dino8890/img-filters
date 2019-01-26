# img-filters

PNG image filters I made as a homework project for a course at my faculty.

It uses Python 3 and CUDA for performing parallel gaussian blur and grayscale filters. It relies on multiprocessing module for CPU parallelism and PyCUDA for interfacing with Nvidia's CUDA.

Feel free to open new issues if you find any or you have a question you want to ask.

## Usage
### Blur filter using GPU

`python3 filter.py source.png dest.png -bf`

### Grayscale filter using CPU

`python3 filter.py source.png dest.png -g`

### Output help text

`python3 filter.py --help`
