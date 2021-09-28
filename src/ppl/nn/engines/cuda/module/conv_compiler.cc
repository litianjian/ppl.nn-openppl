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

#include "ppl/nn/engines/cuda/module/conv_compiler.h"
#include "ppl/nn/engines/cuda/module/cuda_compiler.h"

using namespace std;
namespace ppl { namespace nn { namespace cuda {

const ppl::common::RetCode ConvCompiler::Compile(ir::Node* node, const OptKernelOptions& options) {

    auto node_id = node->GetId();
    auto opt_kerenl = options.info->kernels.find(node_id)->second.get();
    
    CudaOptKernel* cuda_kernel = static_cast<CudaOptKernel*>(opt_kerenl);
    
    auto param = cuda_kernel->GetCommparam();
    CudaCommonParam *cuda_param = static_cast<CudaCommonParam*>(param);
    // cuda_param->module = (void*)
    
    CUDAModuleWrapper *wrapper = new CUDAModuleWrapper();
    CUDAModule *module = new CUDAModule();
    std::string func_name = "nv2spk";
    std::string code;
    std::vector<const char*> params;
    auto ptx = CUDANVRTCCompile(source, params);
    module->SetSourceCode(make_pair<string, string>(move(func_name), move(ptx));
    wrapper->Init(module, func_name, options.device);
    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::cuda