import tvm
from tvm.contrib import graph_executor

model = "bnn_cifar100_BinaryDenseNet28_b200_mxn256x256_inp200x32x32x3"

target = "llvm"
dev = tvm.device(target, 0)
lib: tvm.runtime.Module = tvm.runtime.load_module(f"models/{model}.so")
m = graph_executor.GraphModule(lib["default"](dev))
print(f"Successfully loaded the model {model}")
