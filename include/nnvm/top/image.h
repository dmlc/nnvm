/*!
 *  Copyright (c) 2018 by Contributors
 * \file image.h
 * \brief Auxiliary param for image primitive.
 */
#ifndef NNVM_TOP_IMAGE_H_
#define NNVM_TOP_IMAGE_H_

#include <dmlc/base.h>
#include <dmlc/parameter.h>
#include <nnvm/tuple.h>

namespace nnvm {
namespace top {

struct ImageDecodeParam : public dmlc::Parameter<ImageDecodeParam> {
  int ratio;
  int dct_method;
  int channels;
  float acceptable_fraction;
  bool fancy_upscaling;
  bool try_recover_truncated;
  TShape input_shape;

  DMLC_DECLARE_PARAMETER(ImageDecodeParam) {
    DMLC_DECLARE_FIELD(ratio)
      .describe("TODO: ");
    DMLC_DECLARE_FIELD(dct_method)
      .describe("Specifies the DCT method to be used while decoding.");
    DMLC_DECLARE_FIELD(channels).set_default(3)
      .describe("Number of channels expected in output.");
    DMLC_DECLARE_FIELD(acceptable_fraction).set_default(1.0)
      .describe("TODO:");
    DMLC_DECLARE_FIELD(fancy_upscaling).set_default(true)
      .describe("fancy_upscaling enabled of not.");
    DMLC_DECLARE_FIELD(try_recover_truncated).set_default(false)
      .describe("try_recover_truncated enabled or not");
    DMLC_DECLARE_FIELD(input_shape)
      .describe("Resolution of input image (fixup)");
  }
};


}  // namespace top
}  // namespace nnvm

#endif  // NNVM_TOP_IMAGE_H_
