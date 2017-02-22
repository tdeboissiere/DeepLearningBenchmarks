# Deep Learning benchmarks

Deep learning benchmarks largely inspired by [vgg-benchmarks](https://github.com/aizvorski/vgg-benchmarks).

We tried to get the most out of each framework (GPU util is at 99% for all scripts) but some optimizations may have been overlooked. Fixes and contributions welcome !

## Maxwell Titan X results

| Framework | Time <sup>[1](#foottime)</sup>|
|:---|:---|
| Keras (Theano backend) | 241.478 ms|
| Keras (TensorFlow backend) | 362.206 ms <sup>[2](#kerasnote)</sup>|
|Tensorflow NHWC no XLA| 365.122 ms|
|Tensorflow NHWC with XLA| 300.424 ms|
|Tensorflow NCHW no XLA| 298.478 ms|
|Tensorflow NCHW with XLA| 294.063 ms|

| Framework | Time <sup>[1](#foottime)</sup>|
|:---|:---|
| Keras (Theano backend) BN <sup>[3](#footBN)</sup> mode 0| 347.546 ms|
| Keras (Theano backend) BN mode 2 <sup>[4](#footBNmode)</sup>| 269.074 ms|
| Keras (TensorFlow backend) mode 0 | 560.938 ms|
| Keras (TensorFlow backend) mode 2 | 504.966 ms|
|Tensorflow NHWC + BN no XLA| 493.235 ms|
|Tensorflow NHWC + BN + XLA| 341.702 ms|
|Tensorflow NHWC + fused BN no XLA| 395.963 ms|
|Tensorflow NHWC + fused BN + XLA| 450.777 ms|
|Tensorflow NCHW + BN no XLA| 3642.178 ms|
|Tensorflow NCHW + BN + XLA| 326.325 ms|
|Tensorflow NCHW + fused BN no XLA| 322.396 ms|
|Tensorflow NCHW + fused BN + XLA| 345.121 ms|

<a name="foottime">1</a>: Mean time for 100 (forward + backward + weight update) trials on a VGG16 network with mini batch size of 16. The timer is started right before the first trial and stopped right after the last trial. The reported time is obtained by dividing this interval by the number of trials.

<a name="kerasnote">2</a>: Note that at the moment, keras uses traditional NHWC tensorflow ordering

<a name="footBN">3</a>: Batch normalization layer applied after convolution layers

<a name="footBNmode">4</a>: Mode 0 = use per-batch statistics to normalize the data, and during testing use running averages computed during the training phase. Mode 2 = use per-batch statistics to normalize the data during training and testing.

### System specs

- Ubuntu 14.04
- Cuda 8.0
- cuDNN 5.1.10
- theano '0.9.0beta1.dev-173eef98360c23d7418bad3a36f5fb938724f05f' (cuda backend)
- tensorflow 1.0.0 (compiled from source with CUDA 8.0 cuDNN 5.1.10 and XLA JIT)
- Keras 1.2.2

## Usage

    python main.py

optional arguments:

      --run_keras           Run keras benchmark
      --run_tensorflow      Run pure tensorflow benchmark
      --batch_size BATCH_SIZE
                            Batch size
      --n_trials N_TRIALS   Number of full iterations (forward + backward +
                            update)
      --use_XLA             Whether to use XLA compiler
      --data_format DATA_FORMAT
                            Tensorflow image format
      --use_bn              Use batch normalization (tf benchmark)
      --use_fused           Use fused batch normalization (tf benchmark)




## Examples

    python main.py --run_keras --keras_backend theano

This will run a keras benchmark with theano backend.

    python main.py --run_tensorflow --data_format NHWC --use_XLA

This will run a pure tensorflow benchmark with `NHWC` image ordering and using XLA compiler as shown in [Using JIT compilation](https://www.tensorflow.org/performance/xla/jit)


## Notes

If running a **keras tensorflow benchmark**, make sure the `~/.keras/keras.json` file is set to `{ "image_dim_ordering": "tf", "epsilon": 1e-07, "floatx": "float32", "backend": "tensorflow" }`

If running a **keras theano benchmark**, make sure the `~/.keras/keras.json` file is set to `{ "image_dim_ordering": "th", "epsilon": 1e-07, "floatx": "float32", "backend": "theano" }`