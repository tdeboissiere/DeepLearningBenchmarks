import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Benchmarks largely inspired by https://github.com/aizvorski/vgg-benchmarks')
    parser.add_argument('--run_keras', action="store_true", help="Run keras benchmark")
    parser.add_argument('--run_tensorflow', action="store_true", help="Run pure tensorflow benchmark")
    parser.add_argument('--batch_size', default=16, type=int, help="Batch size")
    parser.add_argument('--n_trials', default=100, type=int,
                        help="Number of full iterations (forward + backward + update)")
    parser.add_argument('--use_XLA', action="store_true", help="Whether to use XLA compilr")
    parser.add_argument('--data_format', default="NHWC", type=str, help="Tensorflow image format")

    args = parser.parse_args()

    assert args.data_format in ["NHWC", "NCHW"]

    if args.run_keras:
        import benchmark_keras
        benchmark_keras.run_VGG16(args.batch_size,
                                  args.n_trials)

    if args.run_tensorflow:
        import benchmark_tensorflow
        benchmark_tensorflow.run_VGG16(args.batch_size,
                                       args.n_trials,
                                       args.data_format,
                                       args.use_XLA)
