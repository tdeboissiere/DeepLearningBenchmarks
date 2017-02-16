# Deep Learning benchmarks

Deep learning benchmarks largely inspired by [https://raw.githubusercontent.com/aizvorski/vgg-benchmarks](vgg-benchmarks).

We tried to get the most out of each framework (GPU util is at 99% for all scripts) but some optimizations may have been overlooked. Fixes and contributions welcome !

## Maxwell Titan X results

| Framework | Time <sup>[1](#foottime)</sup>|
|:---|:---|
| Keras (TensorFlow backend) | 362.206 ms <sup>[2](#kerasnote)</sup>|  
| Keras (Theano backend) | 241.478 ms|
|Tensorflow NHWC no XLA| 365.122 ms|
|Tensorflow NHWC with XLA| 300.424 ms|
|Tensorflow NCHW no XLA| 298.478 ms|
|Tensorflow NCHW with XLA| 294.063 ms|

<a name="foottime">1</a>: Mean time for 100 (forward + backward + weight update) trials on a VGG16 network with mini batch size of 16. The timer is started right before the first trial and stopped right after the last trial. The reported time is obtained by dividing this interval by the number of trials.

<a name="kerasnote">2</a>: Note that at the moment, keras uses traditional NHWC tensorflow ordering

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

    --run_keras           Run keras benchmarks
    --batch_size BATCH_SIZE Batch size (default=16)
    --n_trials N_TRIALS   Number of full iterations (forward + backward + update) (default=100)


## Example

    python main.py --run_keras --keras_backend theano

This will run a keras benchmark with theano backend.


## Notes

If running a **keras tensorflow benchmark**, make sure the `~/.keras/keras.json` file is set to `{ "image_dim_ordering": "tf", "epsilon": 1e-07, "floatx": "float32", "backend": "tensorflow" }`

If running a **keras theano benchmark**, make sure the `~/.keras/keras.json` file is set to `{ "image_dim_ordering": "th", "epsilon": 1e-07, "floatx": "float32", "backend": "theano" }`