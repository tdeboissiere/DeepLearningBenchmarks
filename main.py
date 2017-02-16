import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Benchmarks largely inspired by https://github.com/aizvorski/vgg-benchmarks')
    parser.add_argument('--run_keras', action="store_true", help="Run keras benchmarks")
    parser.add_argument('--batch_size', default=16, type=int, help="Batch size")
    parser.add_argument('--n_trials', default=100, type=int,
                        help="Number of full iterations (forward + backward + update)")

    args = parser.parse_args()

    if args.run_keras:
        import benchmark_keras
        benchmark_keras.run_VGG16(args.batch_size, args.n_trials)
