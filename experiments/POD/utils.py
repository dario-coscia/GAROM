def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='POD + Interpolation tests')
    parser.add_argument('--singular_values', type=int, default=0,
                        help='do you want to print the singular values?')

    args = parser.parse_args()
    return args
