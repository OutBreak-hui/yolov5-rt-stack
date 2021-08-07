# Copyright (c) 2021, The Open PPL teams.
# Copyright (c) 2021, Zhiqiang Wang.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import PurePath, Path
import argparse
import logging
import random
import numpy as np
from pyppl import nn as pplnn, common as pplcommon


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--version", dest="display_version", action="store_true")
    # X86
    parser.add_argument("--use_x86", action="store_true")
    parser.add_argument("--disable_avx512", action="store_true")
    # CUDA
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--quick_select", action="store_true")
    parser.add_argument("--device_id", type=int, default=0,
                        help="Specify the device id to be used.")

    parser.add_argument("--onnx_model", required=True,
                        help="Path of the onnx model file")
    parser.add_argument("--in_shapes", type=str, default=None,
                        help="Shapes of input tensors. dims are separated "
                             "by underline, inputs are separated by comma. "
                             "example: 1_3_128_128, 2_3_400_640, 3_3_768_1024")

    parser.add_argument("--inputs", type=str, default=None,
                        help="The input files are separated by comma.")
    parser.add_argument("--reshaped_inputs", type=str, default=None,
                        help="Binary input files separated by comma. file name "
                             "format: 'name-dims-datatype.dat'. for example: "
                             "input1-1_1_1_1-fp32.dat, input2-1_1_1_1-fp16.dat or "
                             "input3-1_1-int8.dat")

    parser.add_argument("--save_input", action="store_true",
                        help="Switch used to save all input tensors to NDARRAY "
                             "format in one file named 'pplnn_inputs.dat'")
    parser.add_argument("--save_inputs", action="store_true",
                        help="Switch used to save separated input tensors to NDARRAY format")
    parser.add_argument("--save_outputs", action="store_true",
                        help="Switch used to save separated output tensors to NDARRAY format")
    parser.add_argument("--output_path", type=str, default=".",
                        help="The directory to save input/output data if '--save_*' "
                             "options are enabled.")
    return parser


G_PPLNN_DATA_TYPE_NUMPY_MAPS = {
    pplcommon.DATATYPE_INT8: np.int8,
    pplcommon.DATATYPE_INT16: np.int16,
    pplcommon.DATATYPE_INT32: np.int32,
    pplcommon.DATATYPE_INT64: np.int64,
    pplcommon.DATATYPE_UINT8: np.uint8,
    pplcommon.DATATYPE_UINT16: np.uint16,
    pplcommon.DATATYPE_UINT32: np.uint32,
    pplcommon.DATATYPE_UINT64: np.uint64,
    pplcommon.DATATYPE_FLOAT16: np.float16,
    pplcommon.DATATYPE_FLOAT32: np.float32,
    pplcommon.DATATYPE_FLOAT64: np.float64,
    pplcommon.DATATYPE_BOOL: bool,
}


G_PPLNN_DATA_TYPE_STR_MAPS = {
    pplcommon.DATATYPE_INT8: "int8",
    pplcommon.DATATYPE_INT16: "int16",
    pplcommon.DATATYPE_INT32: "int32",
    pplcommon.DATATYPE_INT64: "int64",
    pplcommon.DATATYPE_UINT8: "uint8",
    pplcommon.DATATYPE_UINT16: "uint16",
    pplcommon.DATATYPE_UINT32: "uint32",
    pplcommon.DATATYPE_UINT64: "uint64",
    pplcommon.DATATYPE_FLOAT16: "fp16",
    pplcommon.DATATYPE_FLOAT32: "fp32",
    pplcommon.DATATYPE_FLOAT64: "fp64",
    pplcommon.DATATYPE_BOOL: "bool",
    pplcommon.DATATYPE_UNKNOWN: "unknown",
}


def register_engines(
    use_x86=True,
    disable_avx512=False,
    use_cuda=False,
    device_id=0,
    quick_select=False,
):
    """Register engines"""
    if use_cuda:
        return register_engines_gpu(device_id=device_id, quick_select=quick_select)
    elif use_x86:
        return register_engines_cpu(disable_avx512=disable_avx512)
    else:
        raise NotImplementedError("Currently not supports this device")


def register_engines_cpu(disable_avx512=False):
    engines = []
    x86_engine = pplnn.X86EngineFactory.Create()
    if not x86_engine:
        raise RuntimeError("Create x86 engine failed.")

    if disable_avx512:
        status = x86_engine.Configure(pplnn.X86_CONF_DISABLE_AVX512)
        if status != pplcommon.RC_SUCCESS:
            raise RuntimeError(
                f"x86 engine Configure() failed: {pplcommon.GetRetCodeStr(status)}")

    engines.append(pplnn.Engine(x86_engine))

    return engines


def register_engines_gpu(device_id=0, quick_select=False):
    engines = []

    cuda_options = pplnn.CudaEngineOptions()
    cuda_options.device_id = device_id

    cuda_engine = pplnn.CudaEngineFactory.Create(cuda_options)
    if not cuda_engine:
        raise RuntimeError("Create cuda engine failed.")

    if quick_select:
        status = cuda_engine.Configure(pplnn.CUDA_CONF_USE_DEFAULT_ALGORITHMS)
        if status != pplcommon.RC_SUCCESS:
            raise RuntimeError(
                f"cuda engine Configure() failed: {pplcommon.GetRetCodeStr(status)}")

    engines.append(pplnn.Engine(cuda_engine))

    return engines


def parse_in_shapes(in_shapes_str):
    shape_strs = in_shapes_str.split(",") if in_shapes_str else []
    ret = []
    for s in shape_strs:
        dims = [int(d) for d in s.split("_")]
        ret.append(dims)
    return ret


def set_input_one_by_one(inputs, in_shapes, runtime):
    input_files = inputs.split(",") if inputs else []
    file_num = len(input_files)
    if file_num != runtime.GetInputCount():
        raise RuntimeError(
            f"input file num[{str(file_num)}] != graph input num[{runtime.GetInputCount()}]")

    for i in range(file_num):
        tensor = runtime.GetInputTensor(i)
        shape = tensor.GetShape()
        np_data_type = G_PPLNN_DATA_TYPE_NUMPY_MAPS[shape.GetDataType()]

        dims = []
        if in_shapes:
            dims = in_shapes[i]
        else:
            dims = shape.GetDims()

        in_data = np.fromfile(input_files[i], dtype=np_data_type).reshape(dims)
        status = tensor.ConvertFromHost(in_data)
        if status != pplcommon.RC_SUCCESS:
            raise RuntimeError(
                f"copy data to tensor[{tensor.GetName()}] failed: {pplcommon.GetRetCodeStr(status)}")


def set_reshaped_inputs_one_by_one(reshaped_inputs, runtime):
    input_files = reshaped_inputs.split(",") if reshaped_inputs else []
    file_num = len(input_files)
    if file_num != runtime.GetInputCount():
        raise RuntimeError(
            f"input file num[{str(file_num)}] != graph input num[{runtime.GetInputCount()}]")

    for i in range(file_num):
        input_file_name = PurePath(input_files[i]).name
        file_name_components = input_file_name.split("-")
        if len(file_name_components) != 3:
            raise ValueError(
                f"invalid input filename[{input_files[i]}] in '--reshaped_inputs'.")

        input_shape_str_list = file_name_components[1].split("_")
        input_shape = [int(s) for s in input_shape_str_list]

        tensor = runtime.GetInputTensor(i)
        shape = tensor.GetShape()
        np_data_type = G_PPLNN_DATA_TYPE_NUMPY_MAPS[shape.GetDataType()]
        in_data = np.fromfile(input_files[i], dtype=np_data_type).reshape(input_shape)
        status = tensor.ConvertFromHost(in_data)
        if status != pplcommon.RC_SUCCESS:
            raise RuntimeError(
                f"copy data to tensor[{tensor.GetName()}] failed: {pplcommon.GetRetCodeStr(status)}")


def set_random_inputs(in_shapes, runtime):
    def generate_random_dims(shape):
        dims = shape.GetDims()
        dim_count = len(dims)
        for i in range(2, dim_count):
            if dims[i] == 1:
                dims[i] = random.randint(128, 641)
                if dims[i] % 2 != 0:
                    dims[i] = dims[i] + 1
        return dims

    rng = np.random.default_rng()
    for i in range(runtime.GetInputCount()):
        tensor = runtime.GetInputTensor(i)
        shape = tensor.GetShape()
        data_type = shape.GetDataType()

        np_data_type = G_PPLNN_DATA_TYPE_NUMPY_MAPS[data_type]
        if np_data_type in [np.float16, np.float32, np.float64]:
            lower_bound = -1.0
            upper_bound = 1.0
        else:
            info = np.iinfo(np_data_type)
            lower_bound = info.min
            upper_bound = info.max

        dims = []
        if in_shapes:
            dims = in_shapes[i]
        else:
            dims = generate_random_dims(shape)

        in_data = (upper_bound - lower_bound) * rng.random(dims, dtype=np_data_type) * lower_bound
        status = tensor.ConvertFromHost(in_data)
        if status != pplcommon.RC_SUCCESS:
            raise RuntimeError(
                f"copy data to tensor[{tensor.GetName()}] failed: {pplcommon.GetRetCodeStr(status)}")


def gen_dims_str(dims):
    if not dims:
        return ""

    s = str(dims[0])
    for i in range(1, len(dims)):
        s = s + "_" + str(dims[i])
    return s


def save_inputs_one_by_one(output_path, runtime):
    for i in range(runtime.GetInputCount()):
        tensor = runtime.GetInputTensor(i)
        shape = tensor.GetShape()
        tensor_data = tensor.ConvertToHost()
        if not tensor_data:
            raise RuntimeError(f"copy data from tensor[{tensor.GetName()}] failed.")

        in_data = np.array(tensor_data, copy=False)
        out_data_path_name = (f"pplnn_input_{i}_{tensor.GetName()}-"
                              f"{gen_dims_str(shape.GetDims())}-"
                              f"{G_PPLNN_DATA_TYPE_STR_MAPS[shape.GetDataType()]}.dat")
        in_data.tofile(output_path / out_data_path_name)


def save_inputs_all_in_one(output_path, runtime):
    out_file_name = output_path / "pplnn_inputs.dat"
    fd = open(out_file_name, mode="wb+")
    for i in range(runtime.GetInputCount()):
        tensor = runtime.GetInputTensor(i)
        tensor_data = tensor.ConvertToHost()
        if not tensor_data:
            raise RuntimeError(f"copy data from tensor[{tensor.GetName()}] failed.")

        in_data = np.array(tensor_data, copy=False)
        fd.write(in_data.tobytes())
    fd.close()


def save_outputs_one_by_one(output_path, runtime):
    for i in range(runtime.GetOutputCount()):
        tensor = runtime.GetOutputTensor(i)
        tensor_data = tensor.ConvertToHost()
        if not tensor_data:
            raise RuntimeError(f"copy data from tensor[{tensor.GetName()}] failed.")

        out_data = np.array(tensor_data, copy=False)
        out_data.tofile(output_path / f"pplnn_output-{tensor.GetName()}.dat")


def calc_bytes(dims, item_size):
    nbytes = item_size
    for d in dims:
        nbytes = nbytes * d
    return nbytes


def logging_info(prefix, i, tensor, shape, dims):
    logging.info(f"{prefix}[{i}]")
    logging.info(f"\tname: {tensor.GetName()}")
    logging.info(f"\tdim(s): {dims}")
    logging.info(f"\ttype: {pplcommon.GetDataTypeStr(shape.GetDataType())}")
    logging.info(f"\tformat: {pplcommon.GetDataFormatStr(shape.GetDataFormat())}")
    byte_excluding_padding = calc_bytes(dims, pplcommon.GetSizeOfDataType(shape.GetDataType()))
    logging.info(f"\tbyte(s) excluding padding: {byte_excluding_padding}")


def print_input_output_info(runtime):
    for i in range(runtime.GetInputCount()):
        tensor = runtime.GetInputTensor(i)
        shape = tensor.GetShape()
        dims = shape.GetDims()
        prefix = "input"
        logging_info(prefix, i, tensor, shape, dims)

    for i in range(runtime.GetOutputCount()):
        tensor = runtime.GetOutputTensor(i)
        shape = tensor.GetShape()
        dims = shape.GetDims()
        prefix = "output"
        logging_info(prefix, i, tensor, shape, dims)


def cli_main():

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    parser = get_args_parser()
    args = parser.parse_args()

    if args.display_version:
        logging.info("PPLNN version: " + pplnn.GetVersionString())

    # Register Engines
    use_x86, use_cuda = args.use_x86, args.use_cuda
    if not (use_x86 + use_cuda == 1):
        raise NotImplementedError("Current only one device can be enabled.")

    engines = register_engines(
        use_x86=use_x86,
        disable_avx512=args.disable_avx512,
        use_cuda=use_cuda,
        device_id=args.device_id,
        quick_select=args.quick_select,
    )

    # Creating a Runtime Builder
    runtime_builder = pplnn.OnnxRuntimeBuilderFactory.CreateFromFile(args.onnx_model, engines)
    if not runtime_builder:
        raise RuntimeError("Create OnnxRuntimeBuilder failed.")

    # Creating a Runtime Instance
    runtime_options = pplnn.RuntimeOptions()
    runtime = runtime_builder.CreateRuntime(runtime_options)
    if not runtime:
        raise RuntimeError("Create Runtime instance failed.")

    # Filling Inputs
    in_shapes = parse_in_shapes(args.in_shapes)

    if args.inputs:
        set_input_one_by_one(args.inputs, in_shapes, runtime)
    elif args.reshaped_inputs:
        set_reshaped_inputs_one_by_one(args.reshaped_inputs, runtime)
    else:
        set_random_inputs(in_shapes, runtime)

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.save_input:
        save_inputs_all_in_one(output_path, runtime)
    if args.save_inputs:
        save_inputs_one_by_one(output_path, runtime)

    status = runtime.Run()
    if status != pplcommon.RC_SUCCESS:
        raise RuntimeError(f"Run() failed: {pplcommon.GetRetCodeStr(status)}")

    status = runtime.Sync()
    if status != pplcommon.RC_SUCCESS:
        raise RuntimeError(f"Run() failed: {pplcommon.GetRetCodeStr(status)}")

    print_input_output_info(runtime)

    if args.save_outputs:
        save_outputs_one_by_one(output_path, runtime)

    logging.info("Run OKAY!")


if __name__ == "__main__":
    cli_main()
