#!/usr/bin/env python3
import argparse
import sys
import timeit

import numpy as np
from PIL import Image

from filters import cpu, gpu
from filters import DEFAULT_STANDARD_DEVIATION
from filters import DEFAULT_FILTER_WIDTH

if __name__ == '__main__':
    program_start = timeit.default_timer()

    parser = argparse.ArgumentParser(
        description='apply grayscale or gaussian blur filter to PNG image'
    )
    parser.add_argument('image_src', help='source image file path')
    parser.add_argument('image_result', help='resulting image file path')
    filter_group = parser.add_mutually_exclusive_group(required=True)
    filter_group.add_argument(
        '-g',
        help='apply greyscale filter',
        action='store_true'
    )
    filter_group.add_argument(
        '-b',
        help='apply blur filter',
        action='store_true'
    )
    parser.add_argument(
        '-f',
        help='use GPU for processing',
        action='store_true'
    )
    blur_group = parser.add_argument_group(
        'blur arguments',
        'optional blur arguments'
    )
    blur_group.add_argument(
        '-d',
        help='standard deviation',
        metavar='deviation',
        default=DEFAULT_STANDARD_DEVIATION,
        required=False
    )
    blur_group.add_argument(
        '-w',
        help='filter width',
        metavar='width',
        required=False
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

        standard_deviation = 0
        try:
            standard_deviation = float(args.d)
        except ValueError as e:
            print(
                'Invalid standard deviation value, continuing with {0}.'.format(
                    DEFAULT_STANDARD_DEVIATION
                )
            )
            standard_deviation = DEFAULT_STANDARD_DEVIATION

        filter_width = 0
        try:
            if args.w:
                filter_width = int(args.w)
            else:
                filter_width = DEFAULT_FILTER_WIDTH
        except ValueError as e:
            print(
                'Invalid filter width value, continuing with {0}.'.format(
                    DEFAULT_FILTER_WIDTH
                )
            )
            filter_width = DEFAULT_FILTER_WIDTH
        finally:
            if filter_width % 2 == 0:
                filter_width = filter_width - 1

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
                dest_array = cpu.blur.apply(
                    source_array,
                    standard_deviation,
                    filter_width
                )

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
