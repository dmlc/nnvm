/*!
 *  Copyright (c) 2017 by Contributors
 * \file nn_common.h
 * \brief Common utilities for nn ops.
 */
#ifndef NNVM_TOP_NN_NN_COMMON_H_
#define NNVM_TOP_NN_NN_COMMON_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <nnvm/layout.h>
#include <nnvm/top/nn.h>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>

namespace nnvm {
namespace top {

#define MAX(a, b) (((a) > (b))?(a):(b))

template<typename ParamType>
inline uint32_t UseBiasNumInputs(const NodeAttrs& attrs) {
  const ParamType& param = get<ParamType>(attrs.parsed);
  return param.use_bias ? 3 : 2;
}

template<typename ParamType>
inline std::vector<std::string> UseBiasListInputNames(const NodeAttrs& attrs) {
  const ParamType& param = nnvm::get<ParamType>(attrs.parsed);
  if (param.use_bias) {
    return {"data", "weight", "bias"};
  } else {
    return {"data", "weight"};
  }
}

/*!
 * \brief Convert shape in src_layout to shape in dst_layout
 * \param src original shape
 * \param src_layout layout of original shape
 * \param dst_layout target layout
 * \return shape in target layout
 */
inline TShape ConvertLayout(TShape src, const Layout& src_layout, const Layout& dst_layout) {
  if (src_layout == dst_layout) {
    return src;
  } else if (!src_layout.defined()) {
    LOG(FATAL) << "cannot convert undefined layout to " << dst_layout;
  } else if (!dst_layout.defined()) {
    LOG(FATAL) << "cannot convert " << src_layout << " to undefined layout";
  }

  CHECK(src_layout.convertible(dst_layout)) << "cannot convert from "
                                            << src_layout << " to " << dst_layout;

  TShape dst(dst_layout.ndim());
  for (size_t i = 0; i < src_layout.ndim(); ++i) {
    Layout::LayoutDim src_dim = src_layout[i];
    if (Layout::is_superdim(src_dim)) {
      int dst_major_pos = dst_layout.indexof(Layout::to_superdim(src_dim));
      int dst_minor_pos = dst_layout.indexof(Layout::to_subdim(src_dim));
      int src_minor_pos = src_layout.indexof(Layout::to_subdim(src_dim));
      int src_factor = src_layout.subsizeof(src_dim);
      int dst_factor = dst_layout.subsizeof(src_dim);

      uint32_t src_dim_size = src[i];
      if (src_minor_pos >= 0) {
        CHECK_EQ(src_factor, src[src_minor_pos]) << "src shape " << src
                                                 << " does not agree with layout " << src_layout;
        src_dim_size *= src_factor;
      }

      dst[dst_major_pos] = src_dim_size;
      if (dst_minor_pos >= 0) {
        CHECK_GT(dst_factor, 0);
        CHECK_LE(dst_factor, src_dim_size) << "Converting " << src
                                           << " from " << src_layout
                                           << " to " << dst_factor
                                           << ": cannot split dimension size of "
                                           << src_dim_size << " by " << dst_factor;
        dst[dst_major_pos] /= dst_factor;
        dst[dst_minor_pos] = dst_factor;
      }
    }
  }
  return dst;
}

/*!
 * \brief Derive padding based on the padding scheme
 * \param padding : Original padding scheme
 * \param kernel : kernel shape
 * \param strides : Stride shape
 * \param inshape : Input Tensor shape
 * \param padding_scheme : padding_scheme tp be used.
 * \param layout : Data layout.
 * \return padding based on the padding_scheme or copy original padding.
 */
inline TShape ConvertPaddingScheme(TShape padding, TShape kernel,
                                   TShape strides, TShape inshape,
                                   int padding_scheme, std::string layout) {
  TShape conv_padding = padding;

  if (padding_scheme) {
    int in_height, in_width, pad_h, pad_w;

    if (layout == "NCHW") {
      in_height = inshape[2];
      in_width = inshape[3];
    } else {
      in_height = inshape[1];
      in_width = inshape[2];
    }

    if (in_height % strides[0] == 0)
      pad_h = MAX(kernel[0] - strides[0], 0);
    else
      pad_h = MAX(kernel[0] - (in_height % strides[0]), 0);

    if (in_width % strides[1] == 0)
      pad_w = MAX(kernel[1] - strides[1], 0);
    else
      pad_w = MAX(kernel[1] - (in_width % strides[1]), 0);

    pad_h = pad_h / 2;
    pad_w = pad_w / 2;

    conv_padding[0] = pad_h;
    conv_padding[1] = pad_w;
  }

  return conv_padding;
}

}  // namespace top
}  // namespace nnvm

#endif  // NNVM_TOP_NN_NN_COMMON_H_
