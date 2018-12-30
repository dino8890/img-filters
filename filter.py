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
    parser.add_argument(
        '-d',
        metavar='src_dump_path',
        help='dump source pixel array to the file'
    )
    parser.add_argument(
        '-D',
        metavar='result_dump_path',
        help='dump resulting pixel array to the file'
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

    if args.d:
        try:
            dump_start = timeit.default_timer()
            with open(args.d, "w") as f:
                f.write(str(source_array))

            dump_end = timeit.default_timer()
            print(
                'Time spent dumping source pixel array:',
                dump_end - dump_start,
                'seconds'
            )
        except OSError as e:
            print(e, file=sys.stderr)

    if args.D:
        try:
            dump_start = timeit.default_timer()
            with open(args.D, 'w') as f:
                f.write(str(dest_array))

            dump_end = timeit.default_timer()
            print(
                'Time spent dumping resulting pixel array:',
                dump_end - dump_start,
                'seconds'
            )
        except OSError as e:
            print(e, file=sys.stderr)

    try:
        save_start = timeit.default_timer()
        Image.fromarray(dest_array).save(args.image_result)

        save_end = timeit.default_timer()
        print('Time spent saving image:', save_end - save_start, 'seconds')
    except OSError as e:
        print(e, file=sys.stderr)

    program_end = timeit.default_timer()
    print('Total time spent running:', program_end - program_start, 'seconds')
