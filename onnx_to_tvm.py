import os
import onnx
import numpy as np
import tvm
import tvm.relay as relay
import argparse
import pandas as pd
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.util import tempdir
import tvm.contrib.graph_runtime as runtime

parser = argparse.ArgumentParser(description='AutoTVM from ONNX checkpoints')
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--n_trials', default=1000, type=int)
args = parser.parse_args()

def get_network(filename, batch_size, in_channels, in_x, in_y):
    onnx_model = onnx.load(filename)
    data = np.random.uniform(-1, 1, size=(batch_size,in_channels,in_x,in_y)).astype("float32")
    shape_dict = {'0' : data.shape}
    sym, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    input_shape  = data.shape
    return sym, params, input_shape

if args.cpu:
    target = 'llvm'
    ctx = tvm.cpu()
else:
    target = tvm.target.cuda()
    ctx    = tvm.gpu()


def execute_local():
    sym, params, data = get_network('resnet/resnet50_1.onnx', 1, 64, 56, 56)
    with relay.build_config(opt_level=1):
        intrp = relay.build_module.create_executor('graph',sym, ctx, target)

    tvm_output = intrp.evaluate(sym)(tvm.nd.array(data.astype("float32")), **params).asnumpy()

    print(tvm_output)


#### DEVICE CONFIG ####
target = tvm.target.cuda()

#### TUNING OPTION ####
network = 'resnet50'
log_file = "logs/%s.log" % network
dtype = 'float32'

tuning_option = {
    'log_filename': log_file,
    'n_trial': args.n_trials,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10),
        runner=autotvm.RPCRunner(
            '1080ti',  # change the device key to your key
            '0.0.0.0', 9190,
            number=20, repeat=3, timeout=4, min_repeat_ms=150)
    ),
}

# You can skip the implementation of this function for this tutorial.
def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=True):

    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " %(i+1, len(tasks))

        tuner_obj = XGBTuner(tsk, loss_type='rank')

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tuner_obj.tune(n_trial=min(n_trial, len(tsk.config_space)),
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)

def tune_and_evaluate(tuning_opt):

    num_layers = 49
    df = pd.read_csv('resnet/layer_info.csv')

    for layer in range(num_layers):
        net_fname = 'resnet/' + network + '_' + str(layer) + '.onnx'

        in_c  = int(df.loc[df.filename==net_fname, 'in_channels'])
        in_x  = int(df.loc[df.filename==net_fname, 'input_spatial_x'])

        out_c = int(df.loc[df.filename==net_fname, 'out_channels'])
        #out_x  = int(df.loc[df.filename==net_fname, 'out_spatial_x'])
        #out_shape = (1, out_c, out_x, out_x)

        # extract workloads from relay program
        print("Extract tasks...")
        net, params, input_shape = get_network(net_fname, 1, in_c, in_x, in_x)
        tasks = autotvm.task.extract_from_program(net, target=target,
                                                params=params, ops=(relay.op.nn.conv2d,))

        # run tuning tasks
        print("Tuning...")
        tune_tasks(tasks, **tuning_opt)

        # compile kernels with history best records
        with autotvm.apply_history_best(log_file):
            print("Compile...")
            with relay.build_config(opt_level=3):
                graph, lib, params = relay.build_module.build(
                    net, target=target, params=params)

            # export library
            tmp      = tempdir()
            filename = "net.tar"
            lib.export_library(tmp.relpath(filename))

            # load parameters
            ctx      = tvm.context(str(target), 0)
            module   = runtime.create(graph, lib, ctx)
            data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
            module.set_input('data', data_tvm)
            module.set_input(**params)

            # evaluate
            print("Evaluate inference time cost...")
            ftimer   = module.module.time_evaluator("run", ctx, number=1, repeat=600)
            prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
            print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
                  (np.mean(prof_res), np.std(prof_res)))

# We do not run the tuning in our webpage server since it takes too long.
# Uncomment the following line to run it by yourself.

tune_and_evaluate(tuning_option)
