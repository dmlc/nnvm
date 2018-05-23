/*!
 *  Copyright (c) 2016 by Contributors
 * \file schedule_factory.cc
 * \brief Support for schedule registry.
 */
#include <tvm/tvm.h>
#include <nnvm/base.h>
#include <nnvm/compiler/schedule_factory.h>

#include "topi/generic/default.h"
#include "topi/generic/injective.h"

#include "topi/cuda/injective.h"
#include "topi/cuda/dense.h"
#include "topi/cuda/pooling.h"
#include "topi/cuda/reduction.h"
#include "topi/cuda/softmax.h"

#include "topi/rocm/dense.h"

#include "topi/x86/bnn.h"
#include "topi/x86/default.h"
#include "topi/x86/injective.h"

namespace nnvm {
namespace compiler {

NNVM_REGISTER_SCHEDULE_FACTORY(injective)
.set_generic_schedule(topi::generic::schedule_injective)
.add_schedule({ "cpu" }, topi::x86::schedule_injective)
.add_schedule({ "cuda", "gpu" }, topi::cuda::schedule_injective);

NNVM_REGISTER_SCHEDULE_FACTORY(softmax)
.set_generic_schedule(topi::generic::default_schedule)
.add_schedule({ "cpu" }, topi::x86::default_schedule)
.add_schedule({ "cuda", "gpu" }, topi::cuda::schedule_softmax);

NNVM_REGISTER_SCHEDULE_FACTORY(dense)
.set_generic_schedule(topi::generic::default_schedule)
.add_schedule({ "cpu" }, topi::x86::default_schedule)
.add_schedule({ "cuda", "gpu" }, topi::cuda::schedule_dense)
.add_schedule({ "rocm" }, topi::rocm::schedule_dense);

NNVM_REGISTER_SCHEDULE_FACTORY(pool)
.set_generic_schedule(topi::generic::default_schedule)
.add_schedule({ "cpu" }, topi::x86::default_schedule)
.add_schedule({ "cuda", "gpu" }, topi::cuda::schedule_pool);

NNVM_REGISTER_SCHEDULE_FACTORY(global_pool)
.set_generic_schedule(topi::generic::default_schedule)
.add_schedule({ "cpu" }, topi::x86::default_schedule)
.add_schedule({ "cuda", "gpu" }, topi::cuda::schedule_global_pool);

NNVM_REGISTER_SCHEDULE_FACTORY(reduce)
.set_generic_schedule(topi::generic::default_schedule_auto_inline)
.add_schedule({ "cpu" }, topi::x86::default_schedule_auto_inline)
.add_schedule({ "cuda", "gpu" }, topi::cuda::schedule_reduce);

}  // namespace compiler
}  // namespace nnvm
