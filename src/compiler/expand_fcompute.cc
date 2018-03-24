/*!
 * Copyright (c) 2018 by Contributors
 * \file expand_fcompute.cc
 * \author Nick Hynes
*/
#include <nnvm/graph.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/graph_attr_types.h>
#include <nnvm/pass.h>
#include <nnvm/compiler/op_attr_types.h>
#include <nnvm/top/nn.h>
#include "./graph_transform.h"
#include "./pattern_util.h"

namespace nnvm {
namespace compiler {

Graph ExpandCompute(nnvm::Graph src) {
  const IndexedGraph& idx = src.indexed_graph();
  const ShapeVector& shape_vec = src.GetAttr<ShapeVector>("shape");
  auto transform = [&](uint32_t nid, const NodePtr& n, std::vector<NodeEntry>* ret) {
    static auto& fcompute = Op::GetAttr<FExpandCompute>("FExpandCompute");
    if (!fcompute.count(n->op())) return false;
    std::vector<TShape> input_shapes;
    for (const NodeEntry& inp : n->inputs) {
      input_shapes.push_back(shape_vec[idx.entry_id(inp)]);
    }
    *ret = fcompute[n->op()](n, n->inputs, input_shapes);
    return true;
  };
  return GraphTransform(src, transform);
}

NNVM_REGISTER_PASS(ExpandCompute)
.set_body(ExpandCompute);

}  // namespace compiler
}  // namespace nnvm
