/*!
 *  Copyright (c) 2017 by Contributors
 * \file reduce.cc
 * \brief reduce operator.
 */
#include <numeric>
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/compiler/op_attr_types.h>
#include <nnvm/compiler/util.h>
#include <nnvm/top/tensor.h>
#include "../op_common.h"
#include "../elemwise_op_common.h"
#include "topi/reduction.h"

namespace nnvm {
namespace top {
using namespace tvm;
using namespace nnvm::compiler;

// reduce
DMLC_REGISTER_PARAMETER(ReduceParam);

inline TShape GetReduceAxes(const uint32_t indim,
                            const TShape& axis,
                            bool exclude) {
  if (axis.ndim() == 0) {
    TShape r_axes(indim);
    std::iota(r_axes.begin(), r_axes.end(), 0);
    return r_axes;
  }

  CHECK_LT(axis[axis.ndim() - 1], indim)
    << "Reduction axis " << axis[axis.ndim() - 1]
    << " exceeds input dimensions " << indim;

  TShape in_axis = axis;
  for (auto& i : in_axis) {
    i = i < 0 ? i + indim : i;
    CHECK_GE(i, 0) << "axis out of bounds in reduce operator";
    CHECK_LT(i, indim) << "axis out of bounds in reduce operator";
  }
  std::sort(in_axis.begin(), in_axis.end());
  if (!exclude) return in_axis;
  TShape r_axis(indim - in_axis.ndim());
  for (unsigned i = 0, j = 0, k = 0; i < indim; ++i) {
    if (i == in_axis[j]) {
        ++j;
        continue;
    }
    r_axis[k++] = i;
  }
  return r_axis;
}

inline TShape ReduceShapeImpl(const TShape& ishape,
                              const TShape& axis,
                              bool keepdims,
                              bool exclude) {
  uint32_t indim = ishape.ndim();
  TShape r_axes = GetReduceAxes(indim, axis, exclude);
  if (!r_axes.ndim()) return ishape;
  if (r_axes.ndim() == indim)
    return TShape(keepdims ? indim : 1);

  if (keepdims) {
    TShape oshape(ishape);
    for (unsigned i = 0, j = 0; i < indim; ++i) {
      if (i != r_axes[j]) continue;
      oshape[i] = 1;
      ++j;
    }
    return oshape;
  }

  TShape oshape(indim - r_axes.ndim());
  for (unsigned i = 0, j = 0, k = 0; i < indim; ++i) {
    if (i == r_axes[j]) {
      ++j;
      continue;
    }
    oshape[k++] = ishape[i];
  }
  return oshape;
}

inline bool ReduceShape(const nnvm::NodeAttrs& attrs,
                        std::vector<TShape>* in_attrs,
                        std::vector<TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  if ((*in_attrs)[0].ndim() == 0) return false;
  const ReduceParam& param = nnvm::get<ReduceParam>(attrs.parsed);
  NNVM_ASSIGN_INPUT_SHAPE(
      attrs, *out_attrs, 0,
      ReduceShapeImpl((*in_attrs)[0], param.axis,
                      param.keepdims, param.exclude));
  return true;
}

inline bool CollapseShape(const nnvm::NodeAttrs& attrs,
                          std::vector<TShape>* in_attrs,
                          std::vector<TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  if ((*in_attrs)[0].ndim() == 1) return false;
  NNVM_ASSIGN_INPUT_SHAPE(attrs, *out_attrs, 0, (*in_attrs)[1]);
  return true;
}

template<typename PType>
inline void AxesParamParser(nnvm::NodeAttrs* attrs) {
  PType param;
  param.Init(attrs->dict);
  std::sort(&param.axis[0], &param.axis[param.axis.ndim()]);
  attrs->parsed = std::move(param);
}

#define NNVM_REGISTER_REDUCE_OP(op)                                     \
  NNVM_REGISTER_OP(op)                                                  \
  .add_argument("data", "Tensor", "The input")                          \
  .add_arguments(ReduceParam::__FIELDS__())                             \
  .set_attr_parser(AxesParamParser<ReduceParam>)                        \
  .set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<ReduceParam>) \
  .set_attr<FInferShape>("FInferShape", ReduceShape)                    \
  .set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)               \
  .set_num_inputs(1)                                                    \
  .set_num_outputs(1)

#define NNVM_REGISTER_COLLAPSE_OP(op)                                     \
  NNVM_REGISTER_OP(collapse_ ## op)                                      \
  .add_argument("data", "Tensor", "The input")                          \
  .add_argument("as", "Tensor", "The reference")                          \
  .add_arguments(ReduceParam::__FIELDS__())                             \
  .set_attr_parser(AxesParamParser<ReduceParam>)                        \
  .set_attr<FInferShape>("FInferShape", CollapseShape)                    \
  .set_attr<FInferType>("FInferType", ElemwiseType<2, 1>)               \
  .set_num_inputs(2)                                                    \
  .set_num_outputs(1)

NNVM_REGISTER_REDUCE_OP(sum)
.describe(R"code(Computes the sum of array elements over given axes.

Example::

  data = [[[1,2],[2,3],[1,3]],
          [[1,4],[4,3],[5,2]],
          [[7,1],[7,2],[7,3]]]

  sum(data, axis=1)
  [[  4.   8.]
   [ 10.   9.]
   [ 21.   6.]]

  sum(data, axis=[1,2])
  [ 12.  19.  27.]

)code" NNVM_ADD_FILELINE)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const ReduceParam& param = nnvm::get<ReduceParam>(attrs.parsed);
    TShape r_axes = GetReduceAxes(inputs[0]->shape.size(),
                                  param.axis, param.exclude);
    auto axis = ShapeToArray(r_axes);
    return Array<Tensor>{
      topi::sum(inputs[0], axis, param.keepdims) };
})
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds){
    const ReduceParam& param = nnvm::get<ReduceParam>(n->attrs.parsed);
    std::ostringstream axis; axis << param.axis;
    return std::vector<NodeEntry>{
      MakeNode("expand_like", n->attrs.name + "_grad",
               {ograds[0], n->inputs[0]},
               {{"axis", axis.str()},
                {"keepdims", std::to_string(param.keepdims)},
                {"exclude", std::to_string(param.exclude)}})
  };
});

NNVM_REGISTER_REDUCE_OP(mean)
.describe(R"code(Computes the mean of array elements over given axes.

Example::

  data = [[[1,2],[2,3],[1,3]],
          [[1,4],[4,3],[5,2]],
          [[7,1],[7,2],[7,3]]]

  mean(data, axis=1) = mean(data, axis=[0, 2], exclude=True)
  [[ 1.333   8.333]
   [ 3.333.  3.]
   [ 7.      2.]]

)code" NNVM_ADD_FILELINE)
.set_attr<FExpandCompute>(
  "FExpandCompute", [](const NodePtr& n,
                       const std::vector<NodeEntry>& inputs,
                       const std::vector<TShape>& input_shapes) {
    const ReduceParam& param = nnvm::get<ReduceParam>(n->attrs.parsed);
    unsigned numel = 1;
    TShape ishape = input_shapes[0];
    TShape r_axes = GetReduceAxes(ishape.ndim(), param.axis, param.exclude);
    for (const int& ax : r_axes) {
      numel *= ishape[ax];
    }
    return std::vector<NodeEntry>{
      MakeNode("__div_scalar__", n->attrs.name + "_div",
          {MakeNode("sum", n->attrs.name + "_sum",
                    {n->inputs[0]}, n->attrs.dict)},
          {{"scalar", std::to_string(numel)}})
    };
});

NNVM_REGISTER_REDUCE_OP(var)
.describe(R"code(Computes the variance of array elements over given axes.

Example::

  data = [[[1,2],[2,3],[1,3]],
          [[1,4],[4,3],[5,2]],
          [[7,1],[7,2],[7,3]]]

  var(data, axis=1) = var(data, axis=[0, 2], exclude=True)
  [[ 0.222   0.222]
   [ 2.888.  0.666]
   [ 0.      0.666]]

)code" NNVM_ADD_FILELINE)
.set_attr<FExpandCompute>(
  "FExpandCompute", [](const NodePtr& n,
                       const std::vector<NodeEntry>& inputs,
                       const std::vector<TShape>& input_shapes) {
    auto rp_keepdims = std::unordered_map<std::string, std::string>(n->attrs.dict);
    rp_keepdims["keepdims"] = "true";
    auto inp = n->inputs[0];
    auto mean = MakeNode("mean", n->attrs.name + "_m1", {inp}, rp_keepdims);
    auto diffs = MakeNode("broadcast_sub", n->attrs.name + "_sub", {inp, mean});
    auto diffs2 = MakeNode("__pow_scalar__", n->attrs.name + "_pow",
                           { diffs }, {{"scalar", "2"}});
    return std::vector<NodeEntry>{
      MakeNode("mean", n->attrs.name + "_m2", { diffs2 }, n->attrs.dict)
    };
});

NNVM_REGISTER_REDUCE_OP(max)
.describe(R"code(Computes the max of array elements over given axes.

)code" NNVM_ADD_FILELINE)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const ReduceParam& param = nnvm::get<ReduceParam>(attrs.parsed);
    TShape r_axes = GetReduceAxes(inputs[0]->shape.size(),
                                  param.axis, param.exclude);
    auto axis = ShapeToArray(r_axes);
    return Array<Tensor>{
      topi::max(inputs[0], axis, param.keepdims) };
})
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds){
    const ReduceParam& param = nnvm::get<ReduceParam>(n->attrs.parsed);
    std::ostringstream axis; axis << param.axis;
    NodeEntry sub0 = MakeNode("expand_like", n->attrs.name + "_grad_sub0",
                             {ograds[0], n->inputs[0]},
                             {{"axis", axis.str()},
                              {"keepdims", std::to_string(param.keepdims)},
                              {"exclude", std::to_string(param.exclude)}});
    NodeEntry sub1 = MakeNode("_max_mask", n->attrs.name + "_grad_sub1",
                              {ograds[0]},
                              {{"axis", axis.str()},
                               {"exclude", std::to_string(param.exclude)}});
    return std::vector<NodeEntry>{
      MakeNode("elemwise_mul", n->attrs.name + "_grad", {sub0, sub1})
    };
});

NNVM_REGISTER_REDUCE_OP(min)
.describe(R"code(Computes the min of array elements over given axes.

)code" NNVM_ADD_FILELINE)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const ReduceParam& param = nnvm::get<ReduceParam>(attrs.parsed);
    TShape r_axes = GetReduceAxes(inputs[0]->shape.size(),
                                  param.axis, param.exclude);
    auto axis = ShapeToArray(r_axes);
    return Array<Tensor>{
      topi::min(inputs[0], axis, param.keepdims) };
})
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds){
    const ReduceParam& param = nnvm::get<ReduceParam>(n->attrs.parsed);
    std::ostringstream axis; axis << param.axis;
    NodeEntry sub0 = MakeNode("expand_like", n->attrs.name + "_grad_sub0",
                              {ograds[0], n->inputs[0]},
                              {{"axis", axis.str()},
                               {"keepdims", std::to_string(param.keepdims)},
                               {"exclude", std::to_string(param.exclude)}});
    NodeEntry sub1 = MakeNode("_min_mask", n->attrs.name + "_grad_sub1",
                              {ograds[0]},
                              {{"axis", axis.str()},
                               {"exclude", std::to_string(param.exclude)}});
    return std::vector<NodeEntry>{
      MakeNode("elemwise_mul", n->attrs.name + "_grad", {sub0, sub1})
    };
});

NNVM_REGISTER_COLLAPSE_OP(sum)
.describe(R"code(Reduces lhs to the shape of rhs via sum)code" NNVM_ADD_FILELINE)
.set_attr<FExpandCompute>(
  "FExpandCompute", [](const NodePtr& n,
                 const std::vector<NodeEntry>& inputs,
                 const std::vector<TShape>& input_shapes) {
    auto ishape = input_shapes[0];
    auto oshape = input_shapes[1];
    CHECK_GE(ishape.ndim(), oshape.ndim());
    std::vector<dim_t> r_axes;
    for (uint32_t i = 0; i < ishape.ndim(); ++i) {
      int ii = ishape.ndim() - i - 1;
      int oi = oshape.ndim() - i - 1;
      if (oi < 0 || ishape[ii] != oshape[oi]) {
        r_axes.push_back(ii);
      }
    }
    if (r_axes.size() == 0) { return std::vector<NodeEntry>{n->inputs[0]}; }
    std::ostringstream ax_oss; ax_oss << TShape(r_axes);
    return std::vector<NodeEntry>{
      MakeNode("sum", n->attrs.name, {inputs[0]}, {{"axis", ax_oss.str()}})
    };
});

}  // namespace top
}  // namespace nnvm
