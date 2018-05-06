/*!
 *  Copyright (c) 2017 by Contributors
 * \file matrix_op.cc
 * \brief Matrix operators
 */
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/top/image.h>
#include "../op_common.h"

namespace nnvm {
namespace top {

// image
DMLC_REGISTER_PARAMETER(ImageDecodeParam);

inline bool ImageDecodeShape(const nnvm::NodeAttrs& attrs,
                     std::vector<TShape> *in_attrs,
                     std::vector<TShape> *out_attrs) {
  const ImageDecodeParam& param = nnvm::get<ImageDecodeParam>(attrs.parsed);
  LOG(INFO) << "ImageDecodeShape Called:"  << param.input_shape << ":" << param.channels;

  TShape oshape(3);
  oshape[0] = param.input_shape[0];
  oshape[1] = param.input_shape[1];
  oshape[2] = param.channels;

  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, 0, oshape);

  return true;
}

inline bool InageDecodeInferType(const NodeAttrs& attrs,
                          std::vector<int>* in_attrs,
                          std::vector<int>* out_attrs) {
  NNVM_ASSIGN_OUTPUT_TYPE(attrs, *out_attrs, 0, static_cast<int>(kUint8));
  return true;
}

NNVM_REGISTER_OP(image_decode)
  .describe(R"doc(Image decoder operation.

``dot``'s To decode input encodec image stream of types jpg, bmp, gif and png:

)doc" NNVM_ADD_FILELINE)
.set_support_level(1)
.set_num_inputs(1)
.set_num_outputs(1)
.add_arguments(ImageDecodeParam::__FIELDS__())
.set_attr_parser(ParamParser<ImageDecodeParam>)
.set_attr<FInferShape>("FInferShape", ImageDecodeShape)
.set_attr<FInferType>("FInferType", InageDecodeInferType)
.set_attr<FCorrectLayout>("FCorrectLayout", ZeroLayout)
.add_argument("image_input", "NDArray-or-Symbol", "Encoded image input");

}  // namespace top
}  // namespace nnvm
