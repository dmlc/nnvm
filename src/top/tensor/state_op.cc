/*!
 *  Copyright (c) 2017 by Contributors
 * \file matrix_op.cc
 * \brief Matrix operators
 */
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/top/tensor.h>
#include "../op_common.h"
#include "../elemwise_op_common.h"

namespace nnvm {
namespace top {

NNVM_REGISTER_OP(assign)
.describe(R"doc(Assign second to the first.

It is also helpful when we want to update tensor inplace
without introducing explicit circle in computation graph.

)doc" NNVM_ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<FMutateInputs>(
  "FMutateInputs", [](const NodeAttrs& attrs) {
    return std::vector<uint32_t>{0};
})
.set_attr<FInferShape>("FInferShape", SameShape)
.set_attr<FInplaceOption>(
  "FInplaceOption", [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{1, 0}};
});

}  // namespace top
}  // namespace nnvm
