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

#include "ppl/nn/engines/cuda/kernels/mmcv/mmcv_modulated_deform_conv2d_kernel.h"

#include "cudakernel/nn/deform_conv.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode MMCVModulatedDeformConv2dKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);
    auto weight = ctx->GetInput<TensorImpl>(1);
    auto offset = ctx->GetInput<TensorImpl>(2);
    auto mask = ctx->GetInput<TensorImpl>(3);
    auto bias = ctx->GetInput<TensorImpl>(4);
    
    auto shape_in0 = ctx->GetInput<TensorImpl>(0)->GetShape();
    auto shape_in1 = ctx->GetInput<TensorImpl>(1)->GetShape();
    auto shape_out = ctx->GetOutput<TensorImpl>(0)->GetShape();

    int64_t size = PPLCUDADeformConvGetBufSize(&shape_in0, &shape_in1, &shape_out);
    BufferDesc tmp_buffer_desc;
    auto status = GetCudaDevice()->AllocTmpBuffer(size, &tmp_buffer_desc);
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "alloc tmp buffer size[" << size << "] for kernel[" << GetName()
                   << "] failed: " << ppl::common::GetRetCodeStr(status);
        return status;
    }
    BufferDescGuard __tmp_buffer_guard(&tmp_buffer_desc, [this](BufferDesc* buffer) -> void {
        GetCudaDevice()->FreeTmpBuffer(buffer);
    });
    
    auto stream = GetStream();
    status = PPLCUDADeformConvForward(
        stream, &output->GetShape(), &input->GetShape(), 
        output->GetBufferPtr(), input->GetBufferPtr(), weight->GetBufferPtr(),
        offset->GetBufferPtr(), mask ? mask->GetBufferPtr() : nullptr, bias->GetBufferPtr(),
        param_->groups, param_->deform_groups, param_->channels, param_->num_output,
        param_->stride[0], param_->stride[1], param_->kernel_size[0], param_->kernel_size[1], 
        param_->padding[0], param_->padding[1], param_->dilation[0], param_->dilation[1],
        mask, tmp_buffer_desc.addr);
    
    return status;

}

}}} // namespace ppl::nn::cuda
