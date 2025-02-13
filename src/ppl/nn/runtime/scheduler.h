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

#ifndef _ST_HPC_PPL_NN_RUNTIME_SCHEDULER_H_
#define _ST_HPC_PPL_NN_RUNTIME_SCHEDULER_H_

#include "ppl/common/retcode.h"
#include "ppl/nn/runtime/profiler.h"

namespace ppl { namespace nn {

class Scheduler {
public:
    struct Options final {
        Options(const ir::GraphTopo* t, const std::vector<nodeid_t>* sn, const std::vector<nodeid_t>* last_consumers,
                std::vector<EdgeObject*>* e2o, std::vector<std::unique_ptr<KernelImpl>>* n2k)
            : topo(t), sorted_nodes(sn), edge_last_consumer(last_consumers), edgeid2object(e2o), nodeid2kernel(n2k) {}
        const ir::GraphTopo* topo;
        const std::vector<nodeid_t>* sorted_nodes;
        const std::vector<nodeid_t>* edge_last_consumer;
        std::vector<EdgeObject*>* edgeid2object;
        std::vector<std::unique_ptr<KernelImpl>>* nodeid2kernel;
    };

public:
    virtual ~Scheduler() {}
    virtual ppl::common::RetCode Init(const Options&) = 0;
    virtual ppl::common::RetCode Run(const std::function<ppl::common::RetCode(KernelImpl*, KernelExecContext*)>&,
                                     Profiler*) = 0;
};

}} // namespace ppl::nn

#endif
