import tvm
from tvm.contrib import graph_executor

import parser

model_name = "bnn_cifar100_BinaryDenseNet28_b200_mxn256x256_inp200x32x32x3"

model_data = parser.parse_model_string(model_name)
print(model_data)

target = "llvm"
dev = tvm.device(target, 0)
lib: tvm.runtime.Module = tvm.runtime.load_module(f"models/{model_name}.so")
m = graph_executor.GraphModule(lib["default"](dev))
print(f"Successfully loaded the model {model_name}.")
