# Deep Learning benchmarks

Deep learning benchmarks largely inspired by [https://raw.githubusercontent.com/aizvorski/vgg-benchmarks](vgg-benchmarks).

We tried to get the most out of each framework but some optimizations may have been overlooked. Fixes and contributions welcome !

## Maxwell Titan X results

| Framework | Time <sup>[1](#foottime)</sup>|
|:---|:---|
| Keras (TensorFlow backend) | 362.206 ms <sup>[2](#kerasnote)</sup>|  
| Keras (Theano backend) | 241.478 ms|

<a name="foottime">1</a>: Mean time for 100 (forward + backward + weight updates) on a VGG16 network with mini batch size of 16

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
    --batch_size BATCH_SIZE
                        Batch size
    --n_trials N_TRIALS   Number of full iterations (forward + backward +
                        update)


## Example

    python main.py --run_keras --keras_backend theano

This will run a keras benchmark with theano backend.


## Notes

If running a **tensorflow benchmark**, make sure the `~/.keras/keras.json` file is set to `{ "image_dim_ordering": "tf", "epsilon": 1e-07, "floatx": "float32", "backend": "tensorflow" }`

If running a **theano benchmark**, make sure the `~/.keras/keras.json` file is set to `{ "image_dim_ordering": "th", "epsilon": 1e-07, "floatx": "float32", "backend": "theano" }`