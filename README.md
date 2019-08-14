# pytorch-onnx-tvm
Want to see how well TVM can optimise your neural network (or some subsection of your neural network) for different bits of hardware but have it specified in PyTorch? This project exports your PyTorch models to ONNX and imports them to TVM for autotuning.

## Generate ONNX files

You can generate individual `ONNX` files for each layer of a ResNet by running:

```bash
python resnet_50_to_onnx.py  
```

This will also export a `csv` file containing information about the input/output/weight sizes of each layer.   

## Autotune individual layers

You first need to set up a tracker (this is your host machine), which will coordinate the autotuning experiments on your target devices.

On the host:
```bash
python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190
```

Then on each target device, run:
```bash
python -m tvm.exec.rpc_server --tracker=[HOST-IP]:9190 --key=[DEVICE-KEY]
```

Where `DEVICE-KEY` is a name you have assigned to the target device; you can then address specific target devices when you start tuning. For example, here is how you would tune layer 5 of ResNet-50 on a HiKey board using the GPU:

```bash
python onnx_to_tvm.py --model='resnet50' --layer_info='resnet/layer_info.csv' --layer='resnet/resnet50_0.onnx' --device_key='hikey' --opencl --n_trials=1000
```

To use an Nvidia GPU, omit the `--opencl` flag. 
