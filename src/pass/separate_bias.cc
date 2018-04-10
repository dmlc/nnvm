/*!
 *  Copyright (c) 2017 by Contributors
 * \file separate_bias.cc
 */
#include <nnvm/pass.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/graph_attr_types.h>
#include <nnvm/compiler/op_attr_types.h>
#include "../compiler/graph_transform.h"

namespace nnvm {
namespace pass {
namespace {

using compiler::FSeparateBias;

Graph SeparateBias(nnvm::Graph src) {
  static auto& fseparate_bias_map = Op::GetAttr<FSeparateBias>("FSeparateBias");
  auto transform = [&](uint32_t nid, const NodePtr& n, std::vector<NodeEntry>* ret) {
    if (n->is_variable()) return false;
    if (fseparate_bias_map.count(n->op())) {
      auto fseparate_bias = fseparate_bias_map[n->op()];
      *ret = fseparate_bias(n);
      return true;
    } else {
      return false;
    }
  };
  return compiler::GraphTransform(std::move(src), transform);
}

NNVM_REGISTER_PASS(SeparateBias)
.describe("Infer the device type of each operator. Insert a copy node when there is cross device copy")
.set_body(SeparateBias)
.set_change_graph(true);
}  // namespace
}  // namespace pass
}  // namespace nnvm

