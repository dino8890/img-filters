#!/usr/bin/env python3
import argparse
import sys
import timeit

import numpy as np
from PIL import Image

from filters import cpu, gpu

if __name__ == '__main__':
    program_start = timeit.default_timer()

    parser = argparse.ArgumentParser(
        description='apply grayscale or gaussian blur filter to PNG image'
    )
    parser.add_argument('image_src', help='source image file path')
    parser.add_argument('image_result', help='resulting image file path')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-g',
        help='apply greyscale filter',
        action='store_true'
    )
    group.add_argument(
        '-b',
        help='apply blur filter',
        action='store_true'
    )
    parser.add_argument(
        '-f',
        help='use GPU for processing',
        action='store_true'
    )
    args = parser.parse_args()

    source_array = None
    dest_array = None

    if args.g or args.b:
        try:
            image = Image.open(args.image_src)
        except FileNotFoundError as e:
            sys.exit(e)

        source_array = np.array(image)  # shape = (height, width, channels)

        filter_start = timeit.default_timer()
        if args.f:
            if args.g:
                try:
                    dest_array = gpu.grayscale.apply(source_array)
                except ValueError as e:
                    sys.exit(e)

            if args.b:
                pass
        else:
            if args.g:
                dest_array = cpu.grayscale.apply(source_array)

            if args.b:
                pass

        filter_end = timeit.default_timer()
        print(
            'Time spent applying filter:',
            filter_end - filter_start,
            'seconds.'
        )

    try:
        save_start = timeit.default_timer()
        Image.fromarray(dest_array).save(args.image_result)

        save_end = timeit.default_timer()
        print('Time spent saving image:', save_end - save_start, 'seconds')
    except OSError as e:
        print(e, file=sys.stderr)

    program_end = timeit.default_timer()
    print('Total time spent running:', program_end - program_start, 'seconds')
