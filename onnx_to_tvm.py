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
parser.add_argument('--model', default='resnet50', type=str)
parser.add_argument('--layer_info', default='resnet/layer_info.csv', help='.csv file generated by one of the *_to_onnx.py scripts')
parser.add_argument('--device_key', default='1080ti',  type=str)
parser.add_argument('--opencl', action='store_true')
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--n_trials', default=1000, type=int)
parser.add_argument('--drop_until', default=0, type=int)
args = parser.parse_args()

if not args.opencl:
    os.environ["CUDA_VISIBLE_DEVICES"]='1'

def get_network(filename, batch_size, in_channels, in_x, in_y):
    onnx_model = onnx.load(filename)
    data = np.random.uniform(-1, 1, size=(batch_size,in_channels,in_x,in_y)).astype("float32")
    shape_dict = {'0' : data.shape}
    sym, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    return sym, params

if args.cpu:
    target = 'llvm'
    ctx = tvm.cpu()
else:
    if args.opencl:
        target = tvm.target.create('opencl -device=mali')
        target_host = 'llvm -target=aarch64-linux-gnu'
    else:
        target = tvm.target.cuda()
        ctx    = tvm.gpu()


# You can skip the implementation of this function for this tutorial.
def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename='tuning.log'):

    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "\t[Task %2d/%2d] " %(i+1, len(tasks))

        tuner_obj = XGBTuner(tsk, loss_type='rank')

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

def tune_and_evaluate():
    df = pd.read_csv(args.layer_info)
    if args.drop_until > 0:
        df = df.drop(list(range(args.drop_until)))

    filenames = df.filename

    for net_fname in filenames:
        print('Tuning: ', net_fname)

        #### TUNING OPTION ####
        log_file = "logs/%s.log" % net_fname
        dtype = 'float32'

        tuning_opt = {
            'log_filename': log_file,
            'n_trial': args.n_trials,

            'measure_option': autotvm.measure_option(
                builder=autotvm.LocalBuilder(timeout=10),
                runner=autotvm.RPCRunner(
                    args.device_key,
                    '0.0.0.0', 9190,
                    number=20, repeat=3, timeout=4, min_repeat_ms=150)
            ),
        }


        in_c  = int(df.loc[df.filename==net_fname, 'in_channels'])
        in_x  = int(df.loc[df.filename==net_fname, 'input_spatial_x'])
        out_c = int(df.loc[df.filename==net_fname, 'out_channels'])

        # extract workloads from relay program
        print("\tExtract tasks...")
        net, params = get_network(net_fname, 1, in_c, in_x, in_x)
        tasks = autotvm.task.extract_from_program(net['main'], target=target, params=params, ops=(relay.op.nn.conv2d,))

        # run tuning tasks
        print("\tTuning...")
        tune_tasks(tasks, **tuning_opt)

        # compile kernels with history best records
        with autotvm.apply_history_best(log_file):
            print("\tCompile...")
            with relay.build_config(opt_level=3):
                graph, lib, params = relay.build_module.build(
                    net, target=target, params=params)

            # export library
            tmp      = tempdir()
            filename = "net.tar"
            lib.export_library(tmp.relpath(filename))

            input_shape = (1,in_c,in_x,in_x)

            # load parameters
            ctx      = tvm.context(str(target), 0)
            module   = runtime.create(graph, lib, ctx)
            data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
            module.set_input('0', data_tvm)
            module.set_input(**params)

            # evaluate
            print("\tEvaluate inference time cost...")
            ftimer   = module.module.time_evaluator("run", ctx, number=1, repeat=600)
            prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
            print("\tMean inference time (std dev): %.2f ms (%.2f ms)" %
                  (np.mean(prof_res), np.std(prof_res)))


tune_and_evaluate()
