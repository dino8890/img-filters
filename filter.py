#!/usr/bin/env python3
import argparse
import sys
import timeit

import numpy as np
import PIL.Image

from filters import cpu

try:
    from filters import gpu
    gpu_import_error = False
except ImportError as e:
    gpu_import_error = e

from filters import DEFAULT_STANDARD_DEVIATION, DEFAULT_FILTER_WIDTH


def main():
    program_start = timeit.default_timer()

    parser = argparse.ArgumentParser(
        description='apply grayscale or gaussian blur filter to PNG image'
    )
    parser.add_argument('image_src', help='source image file path')
    parser.add_argument('image_result', help='resulting image file path')
    filter_group = parser.add_mutually_exclusive_group(required=True)
    filter_group.add_argument(
        '-g',
        help='apply grayscale filter',
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

    result_array = None
    if args.g or args.b:
        try:
            image = PIL.Image.open(args.image_src)
            source_array = np.array(image)  # shape = (height, width, channels)
        except FileNotFoundError as e:
            sys.exit(e)

        try:
            standard_deviation = float(args.d)
        except ValueError:
            print(
                ('Invalid standard deviation value, '
                 'continuing with {0}.').format(
                    DEFAULT_STANDARD_DEVIATION
                ),
                file=sys.stderr
            )
            standard_deviation = DEFAULT_STANDARD_DEVIATION

        try:
            if args.w:
                filter_width = int(args.w)
            else:
                filter_width = DEFAULT_FILTER_WIDTH
        except ValueError:
            print(
                'Invalid filter width value, continuing with {0}.'.format(
                    DEFAULT_FILTER_WIDTH
                ),
                file=sys.stderr
            )
            filter_width = DEFAULT_FILTER_WIDTH
        finally:
            if filter_width % 2 == 0:
                filter_width = filter_width - 1

        filter_start = timeit.default_timer()
        if args.f and not gpu_import_error:
            try:
                if args.g:
                    result_array = gpu.grayscale.apply(source_array)
                else:
                    result_array = gpu.blur.apply(
                        source_array,
                        standard_deviation,
                        filter_width
                    )
            except ValueError as e:
                sys.exit(e)
        else:
            if args.f:
                print(
                    'Warning: GPU capabilities disabled, defaulting to CPU.',
                    '{0}'.format(gpu_import_error),
                    sep='\n',
                    file=sys.stderr
                )

            if args.g:
                result_array = cpu.grayscale.apply(source_array)

            if args.b:
                result_array = cpu.blur.apply(
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
        PIL.Image.fromarray(result_array).save(args.image_result)

        save_end = timeit.default_timer()
        print('Time spent saving image:', save_end - save_start, 'seconds')
    except OSError as e:
        print(e, file=sys.stderr)

    program_end = timeit.default_timer()
    print('Total time spent running:', program_end - program_start, 'seconds')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
