import argparse
import utils
import os

# Disable Tensorflow's INFO and WARNING messages
# See http://stackoverflow.com/questions/35911252
if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Benchmarks largely inspired by https://github.com/aizvorski/vgg-benchmarks')
    parser.add_argument('--run_keras', action="store_true", help="Run keras benchmark")
    parser.add_argument('--run_tensorflow', action="store_true", help="Run pure tensorflow benchmark")
    parser.add_argument('--run_pytorch', action="store_true", help="Run pytorch benchmark")
    parser.add_argument('--batch_size', default=16, type=int, help="Batch size")
    parser.add_argument('--n_trials', default=100, type=int,
                        help="Number of full iterations (forward + backward + update)")
    parser.add_argument('--use_XLA', action="store_true", help="Whether to use XLA compiler")
    parser.add_argument('--data_format', default="NCHW", type=str, help="Image format")
    parser.add_argument('--use_bn', action="store_true",help="Use batch normalization (tf benchmark)")
    parser.add_argument('--use_fused', action="store_true",help="Use fused batch normalization (tf benchmark)")

    args = parser.parse_args()

    assert args.data_format in ["NHWC", "NCHW"]

    if args.run_keras:
        import benchmark_keras
        utils.print_module("Running %s..." % benchmark_keras.__name__)
        utils.print_dict(args.__dict__)
        benchmark_keras.run_VGG16(args.batch_size,
                                  args.n_trials,
                                  args.use_bn,
                                  args.data_format)

        # import benchmark_keras
        # utils.print_module("Running %s..." % benchmark_keras.__name__)
        # utils.print_dict(args.__dict__)
        # benchmark_keras.run_SimpleCNN(args.batch_size)

    if args.run_tensorflow:
        import benchmark_tensorflow
        utils.print_module("Running %s..." % benchmark_tensorflow.__name__)
        utils.print_dict(args.__dict__)
        benchmark_tensorflow.run_VGG16(args.batch_size,
                                       args.n_trials,
                                       args.data_format,
                                       args.use_XLA,
                                       args.use_bn,
                                       args.use_fused)

    if args.run_pytorch:
        import benchmark_pytorch

        utils.print_module("Running %s..." % benchmark_pytorch.__name__)
        utils.print_dict(args.__dict__)
        benchmark_pytorch.run_VGG16(args.batch_size,
                                        args.n_trials,)


        # utils.print_module("Running %s..." % benchmark_pytorch.__name__)
        # utils.print_dict(args.__dict__)
        # benchmark_pytorch.run_SimpleCNN(args.batch_size,
        #                                 args.n_trials,)
