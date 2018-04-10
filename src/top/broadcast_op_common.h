/*!
 *  Copyright (c) 2017 by Contributors
 * \file broadcast_op_common.h
 * \brief Common broadcast operator utilities
 */
#ifndef NNVM_TOP_BROADCAST_OP_COMMON_H_
#define NNVM_TOP_BROADCAST_OP_COMMON_H_

#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/top/tensor.h>
#include "./op_common.h"
#include "./elemwise_op_common.h"

namespace nnvm {
namespace top {

// binary broadcast op
inline bool BinaryBroadcastShape(const nnvm::NodeAttrs& attrs,
                                 std::vector<TShape>* in_attrs,
                                 std::vector<TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  const TShape& lhs = (*in_attrs)[0];
  const TShape& rhs = (*in_attrs)[1];

  // avoid pre-mature shape inference.
  if (lhs.ndim() == 0 || rhs.ndim() == 0) return false;

  if (lhs == rhs) {
    NNVM_ASSIGN_INPUT_SHAPE(attrs, *out_attrs, 0, lhs);
    return true;
  }
  TShape out(std::max(lhs.ndim(), rhs.ndim()));
  dim_t bl = out.ndim() - lhs.ndim();
  dim_t br = out.ndim() - rhs.ndim();
  for (dim_t i = 0; i < out.ndim(); ++i) {
    dim_t l = 1, r = 1;
    if (i >= bl) l = lhs[i - bl];
    if (i >= br) r = rhs[i - br];
    if (l != r) {
      if (l == 0 || r == 0) {
        out[i] = 0;
      } else {
        CHECK(l == 1 || r == 1)
          << "operands could not be broadcast together with shapes "
          << lhs << " " << rhs;
        out[i] = std::max(l, r);
      }
    } else {
      out[i] = l;
    }
  }
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, 0, out);
  return true;
}


#define NNVM_REGISTER_BINARY_BROADCAST_OP(name)                     \
  NNVM_REGISTER_OP(name)                                            \
  .set_num_inputs(2)                                                \
  .set_num_outputs(1)                                               \
  .set_attr<FInferShape>("FInferShape", BinaryBroadcastShape)       \
  .set_attr<FInferType>("FInferType", ElemwiseType<2, 1>)           \
  .set_attr<FInplaceOption>("FInplaceOption",                       \
    [](const NodeAttrs& attrs) {                                    \
      return std::vector<std::pair<int, int> >{{0, 0}, {1, 0}};     \
    })                                                              \
  .set_attr<FTVMCompute>(                                           \
    "FTVMCompute", [](const NodeAttrs& attrs,                       \
      const Array<Tensor>& inputs,                                  \
      const Array<Tensor>& out_info) {                              \
        return Array<Tensor>{                                       \
          topi::name(inputs[0], inputs[1]) };                       \
    })                                                              \
  .add_argument("lhs", "Tensor", "first input")                     \
  .add_argument("rhs", "Tensor", "second input")

}  // namespace top
}  // namespace nnvm
#endif  // NNVM_TOP_BROADCAST_OP_COMMON_H_
