/*!
 *  Copyright (c) 2017 by Contributors
 * \file quantized_ops.cc
 * \brief Quantization operators.
 */
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/top/nn.h>
#include <nnvm/top/tensor.h>
#include <nnvm/compiler/op_attr_types.h>
#include "./nn_common.h"
#include "../op_common.h"
#include "../elemwise_op_common.h"
#include "../broadcast_op_common.h"

#define USE_STOCHASTIC_SHIFT 0

namespace nnvm {
namespace top {

using compiler::FQuantize;
using compiler::FSeparateBias;
using compiler::storage_type;
using compiler::accumulate_type;
using compiler::storage_bit;
using compiler::accumulate_bit;
using compiler::input_bit;
using compiler::output_bit;

inline NodeEntry MakeRightShiftNode(NodeEntry n, const std::string& node_name, int valid_bit, int shift_bit) {
  NodeEntry shift;
  if (USE_STOCHASTIC_SHIFT) {
    int limit = (std::pow(2, valid_bit) - 1) - (std::pow(2, shift_bit) - 1);
    LOG(INFO) << "stochastic round limit: " << limit;
    NodeEntry clip = MakeNode("clip", node_name + "_clip",
      {n}, {{"a_min", std::to_string(-limit)}, {"a_max", std::to_string(limit)}});
    NodeEntry round = MakeNode("stochastic_round", node_name + "_round",
      {clip}, {{"bit", std::to_string(shift_bit)}});
    shift = MakeNode("right_shift", node_name + "_rshift",
      {round}, {{"bit", std::to_string(shift_bit)}});
  } else {
    shift = MakeNode("right_shift", node_name + "_rshift",
      {n}, {{"bit", std::to_string(shift_bit)}});
  }
  return shift;
}

inline FQuantize DefaultQuantize(const char* op_name) {
  // identity, reshape, flatten, max_pool2d
  return [=] (uint32_t nid,
              const NodePtr& n,
              const IndexedGraph& idx,
              const std::vector<int>& scale_map,
              const std::vector<int>& repr_bit_map,
              std::vector<int>* out_repr_bit) {
    NodeEntry qnode = MakeNode(op_name, n->attrs.name,
      n->inputs, n->attrs.dict);

    CHECK_EQ(n->num_outputs(), 1);
    out_repr_bit->at(0) = repr_bit_map[idx.entry_id(idx[nid].inputs[0])];
    return qnode.node;
  };
}


inline FQuantize AdditionQuantize(const char* op_name) {
  // elemwise_add, broadcast_add
  // note: left shift and right shift operands to minimize the gap of magnitude
  // TODO: consider data and weight
  return [=] (uint32_t nid,
              const NodePtr& n,
              const IndexedGraph& idx,
              const std::vector<int>& scale_map,
              const std::vector<int>& repr_bit_map,
              std::vector<int>* out_repr_bit) {
    std::vector<NodeEntry> inputs(n->inputs);

    for (size_t i = 0; i < inputs.size(); ++i) {
      if (inputs[i].node->op()->name == "quantize") {
        inputs[i] = MakeNode("cast", inputs[i].node->attrs.name + "_cast",
          {inputs[i]}, {{"dtype", accumulate_type}});
      }
    }

    int arepr_bit = repr_bit_map[idx.entry_id(idx[nid].inputs[0])];
    int brepr_bit = repr_bit_map[idx.entry_id(idx[nid].inputs[1])];

    int gap_bit = std::abs(arepr_bit - brepr_bit);

    // index of left shift node
    // when arepr_bit > brepr_bit is true, lnode_idx = 0, lshift the lhs;
    // when arepr_bit > brepr_bit is false,  lnode_idx = 1, lshift the rhs;
    int lnode_idx = 1 - int(arepr_bit > brepr_bit);
    int rnode_idx = 1 - lnode_idx;

    int lscale_bit = scale_map[idx.entry_id(idx[nid].inputs[lnode_idx])];
    int rscale_bit = scale_map[idx.entry_id(idx[nid].inputs[rnode_idx])];

    int lrepr_bit = repr_bit_map[idx.entry_id(idx[nid].inputs[lnode_idx])];
    int rrepr_bit = repr_bit_map[idx.entry_id(idx[nid].inputs[rnode_idx])];

    int lvalid_bit = lscale_bit - lrepr_bit;
    int rvalid_bit = rscale_bit - rrepr_bit;

    int lshift_bit = 0;
    int rshift_bit = 0;

    int lavaliable_bit = (accumulate_bit - 1) - (lvalid_bit);
    CHECK_GE(lavaliable_bit, 1);

    if (gap_bit <= (lavaliable_bit - 1)) {  // minus one to avoid overflow
      // only need to do left shift without to sacrifice the precision
      // take the rshift node as base magnitude
      lshift_bit = gap_bit;
      out_repr_bit->at(0) = rrepr_bit;
    } else {
      // also need to do right shift
      lshift_bit = (lavaliable_bit - 1);
      rshift_bit = gap_bit - lshift_bit;
      CHECK(rshift_bit < 31);
      out_repr_bit->at(0) = rrepr_bit + rshift_bit;
    }

    // left shift
    if (lshift_bit != 0) {
      inputs[lnode_idx] = MakeNode("left_shift", n->inputs[lnode_idx].node->attrs.name + "_lshift",
        {inputs[lnode_idx]}, {{"bit", std::to_string(lshift_bit)}});
    }

    // right shift
    if (rshift_bit != 0) {
      inputs[rnode_idx] = MakeRightShiftNode(inputs[rnode_idx], n->inputs[rnode_idx].node->attrs.name + "_rshift",
        rvalid_bit, rshift_bit);
    }

    NodeEntry out = MakeNode(op_name, n->attrs.name, inputs);
    return out.node;
  };
}


inline FQuantize MultiplicationQuantize(const char* op_name) {
  // elemwise_mul, broadcast_mul
  return [=] (uint32_t nid,
              const NodePtr& n,
              const IndexedGraph& idx,
              const std::vector<int>& scale_map,
              const std::vector<int>& repr_bit_map,
              std::vector<int>* out_repr_bit) {
    std::string node_name = n->attrs.name;

    NodeEntry lhs = n->inputs[0];
    NodeEntry rhs = n->inputs[1];
    std::string lname = lhs.node->attrs.name;
    std::string rname = rhs.node->attrs.name;

    // unify the data type to accumulate_type
    if (lhs.node->op()->name == "quantize") {
      lhs = MakeNode("cast", lname + "_cast",
        {lhs}, {{"dtype", accumulate_type}});
    }
    if (rhs.node->op()->name == "quantize") {
      rhs = MakeNode("cast", rname + "_cast",
        {rhs}, {{"dtype", accumulate_type}});
    }

    int lrepr_bit = repr_bit_map[idx.entry_id(idx[nid].inputs[0])];
    int rrepr_bit = repr_bit_map[idx.entry_id(idx[nid].inputs[1])];
    out_repr_bit->at(0) = lrepr_bit + rrepr_bit;

    NodeEntry qnode = MakeNode(op_name, node_name + "_quantized",
      {lhs, rhs}, n->attrs.dict);

    return qnode.node;
  };
}


inline FQuantize AccumulationQuantize(const char* op_name) {
  return [=] (uint32_t nid,
              const NodePtr& n,
              const IndexedGraph& idx,
              const std::vector<int>& scale_map,
              const std::vector<int>& repr_bit_map,
              std::vector<int>* out_repr_bit) {
    int reserve_bit = 7; // reserved bit for overflow during accumulation
    std::string node_name = n->attrs.name;

    int in_repr_bit = repr_bit_map[idx.entry_id(idx[nid].inputs[0])];
    int scale_bit = scale_map[idx.entry_id(idx[nid].inputs[0])];
    int valid_bit = scale_bit - in_repr_bit;
    int avaliable_bit = (accumulate_bit - 1) - (valid_bit);
    if (avaliable_bit < reserve_bit) {
      int shift_bit = reserve_bit - avaliable_bit;
      NodeEntry shift = MakeRightShiftNode(n->inputs[0], node_name + "_rshift", valid_bit, shift_bit);
      NodeEntry out = MakeNode(op_name, node_name, {shift}, n->attrs.dict);
      out_repr_bit->at(0) = repr_bit_map[idx.entry_id(idx[nid].inputs[0])] + shift_bit;
      return out.node;
    } else {
      NodeEntry out = MakeNode(op_name, node_name, {n->inputs[0]}, n->attrs.dict);
      out_repr_bit->at(0) = repr_bit_map[idx.entry_id(idx[nid].inputs[0])];
      return out.node;
    }
  };
};


inline FQuantize ComplicatedOpQuantize(const char* op_name) {
  // note: for compilicated operation like conv2d inputs
  // need to be checked and requantize to low-bit representation
  return [=] (uint32_t nid,
              const NodePtr& n,
              const IndexedGraph& idx,
              const std::vector<int>& scale_map,
              const std::vector<int>& repr_bit_map,
              std::vector<int>* out_repr_bit) {
    std::string node_name = n->attrs.name;

    std::vector<NodeEntry> inputs;
    int repr_bit = 0;
    for (size_t i = 0; i < n->inputs.size(); ++i) {
      const auto& e = n->inputs[i];
      int in_repr_bit = repr_bit_map[idx.entry_id(idx[nid].inputs[i])];

      if (e.node->op()->name != "quantize") {
        // requantize the previous output to low-bit
        int scale_bit = scale_map[idx.entry_id(idx[nid].inputs[i])];
        int valid_bit = scale_bit - in_repr_bit;
        int shift_bit = valid_bit - (input_bit - 1);

        if (shift_bit > 0) {
          // do right shift and cast
          NodeEntry shift = MakeRightShiftNode(e, node_name, valid_bit, shift_bit);
          NodeEntry cast = MakeNode("cast", node_name + "_cast2idtype",
            {shift}, {{"dtype", storage_type}});
          inputs.push_back(cast);
          repr_bit += scale_bit - (input_bit - 1);
        } else {
          NodeEntry cast = MakeNode("cast", node_name + "_cast2idtype",
            {e}, {{"dtype", storage_type}});
          inputs.push_back(cast);
          repr_bit += in_repr_bit;
        }
      }
    }
    out_repr_bit->at(0) = repr_bit;

    NodeEntry qnode = MakeNode(op_name, node_name + "_quantized",
      inputs, n->attrs.dict);
    return qnode.node;
  };
};


// quantize

struct QuantizeParam : public dmlc::Parameter<QuantizeParam> {
  int repr_bit;
  int out_type;

  DMLC_DECLARE_PARAMETER(QuantizeParam) {
    DMLC_DECLARE_FIELD(repr_bit);
    DMLC_DECLARE_FIELD(out_type)
    .add_enum("int8", kInt8)
    .add_enum("int16", kInt16)
    .add_enum("int32", kInt32);
  };
};

DMLC_REGISTER_PARAMETER(QuantizeParam);

inline bool QuantizeType(const nnvm::NodeAttrs& attrs,
                         std::vector<int>* in_type,
                         std::vector<int>* out_type) {
  const QuantizeParam& param = nnvm::get<QuantizeParam>(attrs.parsed);
  CHECK_EQ(out_type->size(), 1U);
  NNVM_ASSIGN_OUTPUT_TYPE(attrs, *out_type, 0, param.out_type);
  return true;
}

NNVM_REGISTER_OP(quantize)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<FInferType>("FInferType", QuantizeType)
.add_argument("data", "Tensor", "The input tensor.")
.add_arguments(QuantizeParam::__FIELDS__())
.set_attr_parser(ParamParser<QuantizeParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<QuantizeParam>);


// dequantize

struct DequantizeParam : public dmlc::Parameter<DequantizeParam> {
  int repr_bit; // quantile value
  DMLC_DECLARE_PARAMETER(DequantizeParam) {
    DMLC_DECLARE_FIELD(repr_bit);
  };
};

DMLC_REGISTER_PARAMETER(DequantizeParam);

inline bool DequantizeType(const nnvm::NodeAttrs& attrs,
                           std::vector<int>* in_type,
                           std::vector<int>* out_type) {
  CHECK_EQ(out_type->size(), 1U);
  NNVM_ASSIGN_OUTPUT_TYPE(attrs, *out_type, 0, 0);
  return true;
}

NNVM_REGISTER_OP(dequantize)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<FInferType>("FInferType", DequantizeType)
.add_argument("data", "Tensor", "The input tensor.")
.add_arguments(DequantizeParam::__FIELDS__())
.set_attr_parser(ParamParser<DequantizeParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<DequantizeParam>);


// stochastic round

struct StochasticRoundParam : public dmlc::Parameter<StochasticRoundParam> {
  int bit;
  DMLC_DECLARE_PARAMETER(StochasticRoundParam) {
    DMLC_DECLARE_FIELD(bit);
  };
};

DMLC_REGISTER_PARAMETER(StochasticRoundParam);

NNVM_REGISTER_ELEMWISE_UNARY_OP(stochastic_round)
.add_arguments(StochasticRoundParam::__FIELDS__())
.set_attr_parser(ParamParser<StochasticRoundParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<StochasticRoundParam>);


// noise_shift

struct NoiseShiftParam : public dmlc::Parameter<NoiseShiftParam> {
  int bit;
  DMLC_DECLARE_PARAMETER(NoiseShiftParam) {
    DMLC_DECLARE_FIELD(bit);
  };
};

DMLC_REGISTER_PARAMETER(NoiseShiftParam);

NNVM_REGISTER_ELEMWISE_UNARY_OP(noise_lshift)
.add_arguments(NoiseShiftParam::__FIELDS__())
.set_attr_parser(ParamParser<NoiseShiftParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<NoiseShiftParam>);


// quantized elemwise_add

NNVM_REGISTER_OP(elemwise_add)
.set_attr<FQuantize>("FQuantize", AdditionQuantize("elemwise_add"));


// quantized broadcast_add

NNVM_REGISTER_OP(broadcast_add)
.set_attr<FQuantize>("FQuantize", AdditionQuantize("broadcast_add"));


// quantized elemwise_mul

NNVM_REGISTER_OP(elemwise_mul)
.set_attr<FQuantize>("FQuantize", MultiplicationQuantize("elemwise_mul"));


// quantized broadcast_mul

NNVM_REGISTER_OP(broadcast_mul)
.set_attr<FQuantize>("FQuantize", MultiplicationQuantize("broadcast_mul"));


// quantized identity

NNVM_REGISTER_OP(identity)
.set_attr<FQuantize>("FQuantize", DefaultQuantize("identity"));


// quantized reshape

NNVM_REGISTER_OP(reshape)
.set_attr<FQuantize>("FQuantize", DefaultQuantize("reshape"));


// quantized flatten

NNVM_REGISTER_OP(flatten)
.set_attr<FQuantize>("FQuantize", DefaultQuantize("flatten"));


// quantized concatenate

NNVM_REGISTER_OP(concatenate)
.set_attr<FQuantize>("FQuantize", AdditionQuantize("concatenate"));


// quantized relu

NNVM_REGISTER_OP(relu)
.set_attr<FQuantize>("FQuantize", DefaultQuantize("relu"));


// quantized dense

struct QuantizedDenseParam : public dmlc::Parameter<QuantizedDenseParam> {
  int units;
  bool use_bias;
  int out_type;

  DMLC_DECLARE_PARAMETER(QuantizedDenseParam) {
    DMLC_DECLARE_FIELD(units).set_lower_bound(1)
    .describe("Number of hidden units of the dense transformation.");
    DMLC_DECLARE_FIELD(use_bias).set_default(true)
    .describe("Whether to use bias parameter");
    DMLC_DECLARE_FIELD(out_type)
    .set_default(kInt32)
    .add_enum("int8", kInt8)
    .add_enum("int16", kInt16)
    .add_enum("int32", kInt32);
  }
  // constants
  static const constexpr int kData = 0;
  static const constexpr int kWeight = 1;
  static const constexpr int kBias = 2;
};

DMLC_REGISTER_PARAMETER(QuantizedDenseParam);

inline bool QuantizedDenseShape(const nnvm::NodeAttrs& attrs,
                                std::vector<TShape>* in_shape,
                                std::vector<TShape>* out_shape) {
  const QuantizedDenseParam& param = nnvm::get<QuantizedDenseParam>(attrs.parsed);
  if (param.use_bias) {
    CHECK_EQ(in_shape->size(), 3U) << "Input:[data, weight, bias]";
  } else {
    CHECK_EQ(in_shape->size(), 2U) << "Input:[data, weight]";
  }
  CHECK_EQ(out_shape->size(), 1U);
  // reverse infer
  if ((*out_shape)[0].ndim() != 0) {
    TShape dshape = (*out_shape)[0];
    dshape[dshape.ndim() - 1] = 0;
    NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, QuantizedDenseParam::kData, dshape);
  }
  dim_t num_inputs = 0;
  if ((*in_shape)[QuantizedDenseParam::kData].ndim() != 0) {
    TShape oshape = (*in_shape)[QuantizedDenseParam::kData];
    num_inputs = oshape[oshape.ndim() - 1];
    oshape[oshape.ndim() - 1] = param.units;
    NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, oshape);
  }
  NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, QuantizedDenseParam::kWeight,
                          TShape({param.units, num_inputs}));
  if (param.use_bias) {
    NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, QuantizedDenseParam::kBias, TShape({param.units}));
  }
  return true;
}

inline bool QuantizedDenseType(const nnvm::NodeAttrs& attrs,
                               std::vector<int>* in_type,
                               std::vector<int>* out_type) {
  const QuantizedDenseParam& param = nnvm::get<QuantizedDenseParam>(attrs.parsed);
  CHECK_EQ(out_type->size(), 1U);
  NNVM_ASSIGN_OUTPUT_TYPE(attrs, *out_type, 0, param.out_type);
  return true;
}

NNVM_REGISTER_OP(quantized_dense)
.add_argument("data", "nD Tensor", "Input data.")
.add_argument("weight", "2D Tensor", "Weight matrix.")
.add_argument("bias", "1D Tensor", "Bias parameter.")
.add_arguments(QuantizedDenseParam::__FIELDS__())
.set_attr_parser(ParamParser<QuantizedDenseParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<QuantizedDenseParam>)
.set_num_outputs(1)
.set_num_inputs(UseBiasNumInputs<QuantizedDenseParam>)
.set_attr<FListInputNames>("FListInputNames", UseBiasListInputNames<QuantizedDenseParam>)
.set_attr<FInferShape>("FInferShape", QuantizedDenseShape)
.set_attr<FInferType>("FInferType", QuantizedDenseType)
.set_support_level(1);

NNVM_REGISTER_OP(dense)
.set_attr<FQuantize>("FQuantize", MultiplicationQuantize("quantized_dense"))
.set_attr<FSeparateBias>("FSeparateBias", [] (const NodePtr& n) {
  const DenseParam& param = nnvm::get<DenseParam>(n->attrs.parsed);
  if (param.use_bias == false) return std::vector<NodeEntry>({NodeEntry{n, 0, 0}});
  std::unordered_map<std::string, std::string> dict = n->attrs.dict;
  dict["use_bias"] = "False";
  NodeEntry node = MakeNode(n->op()->name.c_str(), n->attrs.name,
    {n->inputs[0], n->inputs[1]}, dict);
  NodeEntry node_with_bias = MakeNode("broadcast_add", n->attrs.name + "_add_bias",
    {node, n->inputs[2]});
  return std::vector<NodeEntry>({node_with_bias});
});


// quantized conv2d

struct QuantizedConv2DParam : public dmlc::Parameter<QuantizedConv2DParam> {
  int channels;
  TShape kernel_size;
  TShape strides;
  TShape padding;
  TShape dilation;
  int groups;
  int layout;
  bool use_bias;
  int out_type;

  DMLC_DECLARE_PARAMETER(QuantizedConv2DParam) {
    DMLC_DECLARE_FIELD(channels)
      .describe("The dimensionality of the output space"
                "i.e. the number of output channels in the convolution.");
    DMLC_DECLARE_FIELD(kernel_size)
      .describe("Specifies the dimensions of the convolution window.");
    DMLC_DECLARE_FIELD(strides).set_default(TShape({1, 1}))
      .describe("Specifies the strides of the convolution.");
    DMLC_DECLARE_FIELD(padding).set_default(TShape({0, 0}))
      .describe("If padding is non-zero, then the input is implicitly zero-padded"
                "on both sides for padding number of points");
    DMLC_DECLARE_FIELD(dilation).set_default(TShape({1, 1}))
      .describe("Specifies the dilation rate to use for dilated convolution.");
    DMLC_DECLARE_FIELD(groups).set_default(1)
      .describe("Controls the connections between inputs and outputs."
                "At groups=1, all inputs are convolved to all outputs."
                "At groups=2, the operation becomes equivalent to having two convolution"
                "layers side by side, each seeing half the input channels, and producing"
                "half the output channels, and both subsequently concatenated.");
    DMLC_DECLARE_FIELD(layout)
      .add_enum("NCHW", kNCHW)
      .add_enum("NHWC", kNHWC)
      .set_default(kNCHW)
      .describe("Dimension ordering of data and weight. Can be 'NCHW', 'NHWC', etc."
                "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
                "dimensions respectively. Convolution is applied on the 'H' and"
                "'W' dimensions.");
    DMLC_DECLARE_FIELD(use_bias).set_default(true)
      .describe("Whether the layer uses a bias vector.");
    DMLC_DECLARE_FIELD(out_type)
    .set_default(kInt32)
    .add_enum("int8", kInt8)
    .add_enum("int16", kInt16)
    .add_enum("int32", kInt32);
  }
  // constants
  static const constexpr int kData = 0;
  static const constexpr int kWeight = 1;
  static const constexpr int kBias = 2;
};


DMLC_REGISTER_PARAMETER(QuantizedConv2DParam);

inline bool QuantizedConv2DShape(const nnvm::NodeAttrs& attrs,
                                 std::vector<TShape>* in_shape,
                                 std::vector<TShape>* out_shape) {
  const QuantizedConv2DParam& param = nnvm::get<QuantizedConv2DParam>(attrs.parsed);
  if (param.use_bias) {
    CHECK_EQ(in_shape->size(), 3U) << "Input:[data, weight, bias]";
  } else {
    CHECK_EQ(in_shape->size(), 2U) << "Input:[data, weight]";
  }
  CHECK_EQ(out_shape->size(), 1U);

  TShape dshape = in_shape->at(0);
  if (dshape.ndim() == 0) return false;
  dshape = ConvertLayout(dshape, param.layout, kNCHW);

  CHECK_EQ(dshape.ndim(), 4U) << "Input data should be 4D";
  CHECK_EQ(param.kernel_size.ndim(), 2U);
  CHECK_EQ(param.strides.ndim(), 2U)
      << "incorrect stride size: " << param.strides;
  CHECK_EQ(param.dilation.ndim(), 2U)
      << "incorrect dilate size: " << param.dilation;
  CHECK_EQ(dshape[1] % param.groups, 0U)
      << "input channels must divide group size";
  CHECK_EQ(param.channels % param.groups, 0U)
      << "output channels must divide group size";

  TShape wshape({param.channels / param.groups,
                 dshape[1] / param.groups,
                 param.kernel_size[0],
                 param.kernel_size[1]});

  wshape = ConvertLayout(wshape, kNCHW, param.layout);
  wshape[0] *= param.groups;

  NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, QuantizedConv2DParam::kWeight, wshape);
  if (param.use_bias) {
    NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape,
                            QuantizedConv2DParam::kBias, TShape({param.channels}));
  }
  // dilation
  dim_t dilated_ksize_y = 1 + (param.kernel_size[0] - 1) * param.dilation[0];
  dim_t dilated_ksize_x = 1 + (param.kernel_size[1] - 1) * param.dilation[1];
  TShape oshape({dshape[0], param.channels, 0, 0});
  if (dshape[2] != 0) {
    oshape[2] = (dshape[2] + param.padding[0] * 2 - dilated_ksize_y) / param.strides[0] + 1;
  }
  if (dshape[3] != 0) {
    oshape[3] = (dshape[3] + param.padding[1] * 2 - dilated_ksize_x) / param.strides[1] + 1;
  }
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0,
                           ConvertLayout(oshape, kNCHW, param.layout));
  // Perform incomplete shape inference. Fill in the missing values in data shape.
  // 1) We can always fill in the batch_size.
  // 2) We can back-calculate the input height/width if the corresponding stride is 1.
  oshape = ConvertLayout((*out_shape)[0], param.layout, kNCHW);
  dshape[0] = oshape[0];
  if (oshape[2] && param.strides[0] == 1) {
    dshape[2] = oshape[2] + dilated_ksize_y - 1 - 2 * param.padding[0];
  }
  if (oshape[3] && param.strides[1] == 1) {
    dshape[3] = oshape[3] + dilated_ksize_x - 1 - 2 * param.padding[1];
  }
  NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, QuantizedConv2DParam::kData,
                          ConvertLayout(dshape, kNCHW, param.layout));
  // Check whether the kernel sizes are valid
  if (dshape[2] != 0) {
    CHECK_LE(dilated_ksize_y, dshape[2] + 2 * param.padding[0])
      << "kernel size exceed input";
  }
  if (dshape[3] != 0) {
    CHECK_LE(dilated_ksize_x, dshape[3] + 2 * param.padding[1])
        << "kernel size exceed input";
  }
  return true;
}

inline bool QuantizedConv2DType(const nnvm::NodeAttrs& attrs,
                                std::vector<int>* in_type,
                                std::vector<int>* out_type) {
  const QuantizedConv2DParam& param = nnvm::get<QuantizedConv2DParam>(attrs.parsed);
  if (param.use_bias) {
    CHECK_EQ(in_type->size(), 3U) << "Input:[data, weight, bias]";
  } else {
    CHECK_EQ(in_type->size(), 2U) << "Input:[data, weight]";
  }
  CHECK_EQ(out_type->size(), 1U);
  NNVM_ASSIGN_OUTPUT_TYPE(attrs, *out_type, 0, param.out_type);
  return true;
}

NNVM_REGISTER_OP(quantized_conv2d)
.add_argument("data", "4D Tensor", "Input data.")
.add_argument("weight", "4D Tensor", "Weight matrix.")
.add_argument("bias", "1D Tensor", "Bias parameter.")
.add_arguments(QuantizedConv2DParam::__FIELDS__())
.set_attr_parser(ParamParser<QuantizedConv2DParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<QuantizedConv2DParam>)
.set_attr<FListInputNames>("FListInputNames", UseBiasListInputNames<QuantizedConv2DParam>)
.set_attr<FInferShape>("FInferShape", QuantizedConv2DShape)
.set_attr<FInferType>("FInferType", QuantizedConv2DType)
.set_num_outputs(1)
.set_num_inputs(UseBiasNumInputs<QuantizedConv2DParam>)
.set_support_level(2);

NNVM_REGISTER_OP(conv2d)
.set_attr<FQuantize>("FQuantize", ComplicatedOpQuantize("quantized_conv2d"))
.set_attr<FSeparateBias>("FSeparateBias", [] (const NodePtr& n) {
  const Conv2DParam& param = nnvm::get<Conv2DParam>(n->attrs.parsed);
  if (param.use_bias == false) return std::vector<NodeEntry>({NodeEntry{n, 0, 0}});
  std::unordered_map<std::string, std::string> dict = n->attrs.dict;
  dict["use_bias"] = "False";
  NodeEntry node = MakeNode(n->op()->name.c_str(), n->attrs.name,
    {n->inputs[0], n->inputs[1]}, dict);
  NodeEntry bias = n->inputs[2];
  NodeEntry expand = MakeNode("expand_dims", bias.node->attrs.name + "_expand",
    {bias}, {{"axis", "1"}, {"num_newaxis", "2"}});
  NodeEntry node_with_bias = MakeNode("broadcast_add", n->attrs.name + "_add_bias",
    {node, expand});
  return std::vector<NodeEntry>({node_with_bias});
});


// quantized max_pool2d

NNVM_REGISTER_OP(max_pool2d)
.set_attr<FQuantize>("FQuantize", DefaultQuantize("max_pool2d"));


// quantized avg_pool2d

NNVM_REGISTER_OP(avg_pool2d)
.set_attr<FQuantize>("FQuantize", AccumulationQuantize("avg_pool2d"));


// quantized global_avg_pool2d

NNVM_REGISTER_OP(global_avg_pool2d)
.set_attr<FQuantize>("FQuantize", AccumulationQuantize("global_avg_pool2d"));


}  // namespace top
}  // namespace nnvm
