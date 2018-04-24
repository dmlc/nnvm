/*!
 *  Copyright (c) 2017 by Contributors
 * \file broadcast.cc
 * \brief broadcast operator.
 */
#include <tvm/expr.h>
#include <tvm/packed_func_ext.h>
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/compiler/op_attr_types.h>
#include <nnvm/compiler/util.h>
#include <nnvm/top/tensor.h>
#include "../op_common.h"
#include "../elemwise_op_common.h"
#include "../broadcast_op_common.h"
#include "topi/broadcast.h"

namespace nnvm {
namespace top {
using namespace tvm;
using namespace nnvm::compiler;

// broadcast_to
DMLC_REGISTER_PARAMETER(BroadcastToParam);

inline bool BroadcastToInferShape(const NodeAttrs& attrs,
                                  std::vector<TShape>* in_attrs,
                                  std::vector<TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  const TShape& ishape = (*in_attrs)[0];
  if (ishape.ndim() == 0) return false;

  const BroadcastToParam& param = nnvm::get<BroadcastToParam>(attrs.parsed);
  CHECK_EQ(ishape.ndim(), param.shape.ndim())
      << "Operand of shape " << ishape
      << " cannot be broadcasted to " << param.shape;
  TShape oshape = param.shape;
  for (dim_t i = 0; i < ishape.ndim(); ++i) {
    if (oshape[i] != 0) {
      CHECK(ishape[i] == oshape[i] || ishape[i] == 1)
        << "Array cannot be broadcasted from " <<
          ishape << " to " << param.shape;
    } else {
      oshape[i] = ishape[i];
    }
  }
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, 0, oshape);
  return true;
}

NNVM_REGISTER_OP(broadcast_to)
.describe(R"code(Broadcasts the input array to a new shape.

Broadcasting is a mechanism that allows NDArrays to perform arithmetic operations
with arrays of different shapes efficiently without creating multiple copies of arrays.
Also see, `Broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_ for more explanation.

Broadcasting is allowed on axes with size 1, such as from `(2,1,3,1)` to
`(2,8,3,9)`. Elements will be duplicated on the broadcasted axes.

For example::

   broadcast_to([[1,2,3]], shape=(2,3)) = [[ 1.,  2.,  3.],
                                           [ 1.,  2.,  3.]])

The dimension which you do not want to change can also be kept as `0` which means copy the original value.
So with `shape=(2,0)`, we will obtain the same result as in the above example.

)code" NNVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Input data.")
.add_arguments(BroadcastToParam::__FIELDS__())
.set_attr_parser(ParamParser<BroadcastToParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<BroadcastToParam>)
.set_attr<FInferShape>("FInferShape", BroadcastToInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
    const Array<Tensor>& inputs,
    const Array<Tensor>& out_info) {
      const BroadcastToParam& param = nnvm::get<BroadcastToParam>(attrs.parsed);
      auto shape = ShapeToArray(param.shape);
      return Array<Tensor>{ topi::broadcast_to(inputs[0], shape) };
  })
.set_num_inputs(1)
.set_num_outputs(1)
.set_support_level(4);


NNVM_REGISTER_BINARY_BROADCAST_OP(broadcast_add)
.add_alias("__add_symbol__")
.describe(R"code(Returns element-wise sum of the input arrays with broadcasting.

Example::

   x = [[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]]

   y = [[ 0.],
        [ 1.]]

   broadcast_add(x, y) = [[ 1.,  1.,  1.],
                          [ 2.,  2.,  2.]]

)code" NNVM_ADD_FILELINE);


NNVM_REGISTER_BINARY_BROADCAST_OP(broadcast_sub)
.add_alias("__sub_symbol__")
.describe(R"code(Returns element-wise difference of the input arrays with broadcasting.

Example::

   x = [[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]]

   y = [[ 0.],
        [ 1.]]

   broadcast_sub(x, y) = [[ 1.,  1.,  1.],
                          [ 0.,  0.,  0.]]

)code" NNVM_ADD_FILELINE);


NNVM_REGISTER_BINARY_BROADCAST_OP(broadcast_mul)
.add_alias("__mul_symbol__")
.describe(R"code(Returns element-wise product of the input arrays with broadcasting.

Example::

   x = [[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]]

   y = [[ 0.],
        [ 1.]]

   broadcast_mul(x, y) = [[ 0.,  0.,  0.],
                          [ 1.,  1.,  1.]]
)code" NNVM_ADD_FILELINE);


NNVM_REGISTER_BINARY_BROADCAST_OP(broadcast_div)
.add_alias("__div_symbol__")
.describe(R"code(Returns element-wise division of the input arrays with broadcasting.

Example::

   x = [[ 6.,  6.,  6.],
        [ 6.,  6.,  6.]]

   y = [[ 2.],
        [ 3.]]

   broadcast_div(x, y) = [[ 3.,  3.,  3.],
                          [ 2.,  2.,  2.]]

)code" NNVM_ADD_FILELINE);

}  // namespace top
}  // namespace nnvm
