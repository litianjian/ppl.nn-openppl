// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include <fstream>
#include <sstream>

#include "ppl/nn/engines/cuda/module/conv_compiler.h"
#include "ppl/nn/engines/cuda/module/cuda_compiler.h"
#include "ppl/nn/engines/cuda/params/conv_extra_param.h"


using namespace std;
namespace ppl { namespace nn { namespace cuda {

const ppl::common::RetCode ConvCompiler::Compile(ir::Node* node, const OptKernelOptions& options) {

    auto node_id = node->GetId();
    auto opt_kerenl = options.info->kernels.find(node_id)->second.get();
    
    CudaOptKernel* cuda_kernel = static_cast<CudaOptKernel*>(opt_kerenl);
    
    auto param = cuda_kernel->GetCommparam();
    CudaCommonParam* cuda_param = static_cast<CudaCommonParam*>(param);

    CudaConvParam* conv_param = static_cast<CudaConvParam*>(cuda_kernel->GetParam());
    

    std::string name = conv_param->extra_param.algo_info.algo_name;
    std::string source = conv_param->extra_param.algo_info.kernel_code;
    struct select_param_t tiles = conv_param->extra_param.algo_info.tiles;

    
    
    // cuda_param->module = (void*)
    std::cout << name << std::endl;
    // std::cout << source << std::endl;
    // file.open(name);
    // std::stringstream buffer;
    // buffer << file.rdbuf();
    // std::string code(buffer.str());
    // size_t begin = code.find("#define KERNEL_NAME");
    // size_t end = code.find("#include <cuda_fp16.h>");
    // auto kernel_name = code.substr(begin + 20, end - begin - 22);
    // std::cout << kernel_name << std::endl;

    std::vector<std::string> compile_params;
    std::vector<const char*> param_cstring{};
    compile_params.push_back("-arch=compute_75");
    compile_params.push_back("--include-path=/usr/local/cuda/include");
    // compile_params.push_back("--include-path=/home/litianjian/workspace/github/ppl.nn-openppl/src/ppl/nn/engines/cuda/impls/src/nn/conv");

    compile_params.push_back("--include-path=/usr/include");
    for (auto &string : compile_params) {
       param_cstring.push_back(string.c_str());
    }
//    std::cout << code << std::endl;
    //Create an instance of nvrtcProgram with the conv code string.
    // nvrtcProgram conv1;
    // PPL_NVRTC_SAFE_CALL(nvrtcCreateProgram(&conv1, code.c_str(), "idxn_b32x16_w32x16_k16_s16.cu", 0, NULL, NULL));
    // (nvrtcCompileProgram(conv1, param_cstring.size(), param_cstring.data()));
    // size_t log_size;
    // (nvrtcGetProgramLogSize(conv1, &log_size));
    // char* log = new char[log_size];
    // (nvrtcGetProgramLog(conv1, log));
    // std::cout<< log << std::endl;
    // delete[] log;
     
    // size_t ptx_size;
    // PPL_NVRTC_SAFE_CALL(nvrtcGetPTXSize(conv1, &ptx_size));
    // char* ptx = new char[ptx_size];
    // std::cout << ptx_size << std::endl;
    // PPL_NVRTC_SAFE_CALL(nvrtcGetPTX(conv1, ptx));
    // // std::cout<<ptx<<std::endl;
    // PPL_NVRTC_SAFE_CALL(nvrtcDestroyProgram(&conv1));
    //Load the generated PTX and get a handle to the conv kernel.
    // CUdevice cu_device;
    // CUcontext context;
    // CUmodule module;
    // CUfunction function;
    // PPL_CUDA_SAFE_CALL(cuInit(0));
    // PPL_CUDA_SAFE_CALL(cuDeviceGet(&cu_device, 0));
    // PPL_RUNTIME_SAFE_CALL(cudaDeviceSynchronize());
    // // PPL_CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cu_device));
    // PPL_CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));
    // PPL_CUDA_SAFE_CALL(cuModuleGetFunction(&function, module, "nv2spkConv_hmma1688_nhwc_fn_b128x128_w64x64_k32_s32_buf2"));


    // std::string ptx_code(ptx);
    CUDAModuleWrapper* wrapper = new CUDAModuleWrapper();
    CUDAModule* cuda_module = new CUDAModule();
    std::string func_name = name;//"nv2spkConv_hmma1688_nhwc_fn_b128x128_w64x64_k32_s32_buf2";
    // cuda_module->SetCuModule(module);
    cuda_param->module = (void*)cuda_module;


    // std::vector<const char*> params;
    // auto source = module->GetSourceCode();
    auto ptx_code = CUDANVRTCCompile(make_pair<string, string>(std::move(name), std::move(source)), param_cstring);
    // std::cout << ptx_code << std::endl;
    cuda_module->SetSourceCode(func_name, ptx_code);
    wrapper->Init(cuda_module, func_name, options.device);

    ModuleMap* module_map = options.cuda_module_manager->GetModule();
    module_map->emplace(make_pair<nodeid_t, CUDAModuleWrapper*>(move(node_id), move(wrapper)));
    // module_map
    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::cuda