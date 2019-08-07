# pytorch-onnx-tvm
PyTorch -> ONNX -> TVM for autotuning

## Generate ONNX files

You can generate individual `ONNX` files for each layer of a ResNet by running:

```bash
python resnet_50_to_onnx.py  
```

This will also export a `csv` file containing information about the input/output/weight sizes of each layer.   

## Autotune individual layers

### Nvidia GPU

On the host:
```bash
python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190
```

On each device run:
```bash
python -m tvm.exec.rpc_server --tracker=0.0.0.0:9190 --key=1080ti
```

Then begin tuning:

```bash
python onnx_to_tvm.py
```

### ARM GPU
Notes coming soon...  
