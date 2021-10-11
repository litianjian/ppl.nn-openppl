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

#include "gene_header.h"

#include <fstream>
#include <sstream>

std::string GeneHeader::Find(const std::string& path) {
    auto header_ref = header_code_.find(path);
    if (header_ref != header_code_.end()) {
        return header_ref->second;
    }
    return "";
}

void GeneHeader::InitIncludeFile(std::string path) {
    std::ifstream os_read;
    // path = "/mnt/hpc/xusi/ppl.jit/src/ppl/nn/engines/cuda/impls/src/nn/conv/" + path;
	std::string path1 = "/home/litianjian/workspace/github/ppl.nn-openppl/src/ppl/nn/engines/cuda/impls/src/nn/conv/" + path;
    os_read.open(path1);
    std::stringstream file_str;
	file_str << os_read.rdbuf();
	header_code_.emplace(path, file_str.str());
    return;
}

GeneHeader::GeneHeader() {
	InitIncludeFile("2spk/common/const_macros.h");
    InitIncludeFile("2spk/f1/bound_macros.h");
    InitIncludeFile("2spk/f3/bound_macros.h");
    InitIncludeFile("2spk/fn/bound_macros.h");
    InitIncludeFile("2spk/fs/bound_macros.h");
    InitIncludeFile("2spk/common/ldsm_macros.h");
    InitIncludeFile("2spk/f1/dmem_macros.h");
    InitIncludeFile("2spk/f3/dmem_macros.h");
    InitIncludeFile("2spk/fn/dmem_macros.h");
    InitIncludeFile("2spk/fs/dmem_macros.h");
    InitIncludeFile("2spk/common/hmma_macros.h");
    InitIncludeFile("2spk/common/reduce_macros.h");
    InitIncludeFile("2spk/common/smem_macros.h");
    InitIncludeFile("2spk/common/output_macros.h");
    InitIncludeFile("2spk/common/main_body.h");
    InitIncludeFile("2spk/common/uni_undefs.h");
    
    InitIncludeFile("idxn/common/const_macros.h");
	InitIncludeFile("idxn/common/dmem_i1_macros.h");
	InitIncludeFile("idxn/common/hmma_i1_macros.h");
	InitIncludeFile("idxn/common/dmem_i2_macros.h");
	InitIncludeFile("idxn/common/hmma_i2_macros.h");
	InitIncludeFile("idxn/common/dmem_i4_macros.h");
	InitIncludeFile("idxn/common/hmma_i4_macros.h");
    InitIncludeFile("idxn/common/output_macros.h");
    InitIncludeFile("idxn/common/main_body.h");
    InitIncludeFile("idxn/common/uni_undefs.h");
}
