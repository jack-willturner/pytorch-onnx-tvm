for l in {0..12}
do
  #python onnx_to_tvm.py --layer_info=vgg/layer_info.csv --layer=vgg/vgg16_$l.onnx --device_key=1080ti
  python onnx_to_tvm.py --log_file=logs/vgg/vgg16_$l.onnx.log  --layer=vgg/vgg16_$l.onnx --layer_info='vgg/layer_info.csv'
done
