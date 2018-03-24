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
  std::unordered_map<std::string, uint32_t> name2idx;
  for (uint32_t i = 0; i < idx.num_nodes(); ++i) {
    name2idx[idx[i].source->attrs.name] = i;
  }
  const ShapeVector& shape_vec = src.GetAttr<ShapeVector>("shape");
  auto transform = [&](uint32_t nid, const NodePtr& n, std::vector<NodeEntry>* ret) {
    static auto& fcompute = Op::GetAttr<FExpandCompute>("FExpandCompute");
    if (!fcompute.count(n->op())) return false;
    std::vector<TShape> input_shapes;
    for (const NodeEntry& inp : n->inputs) {
      input_shapes.push_back(shape_vec[name2idx[inp.node->attrs.name]]);
    }
    std::vector<NodeEntry> exp = fcompute[n->op()](n, n->inputs, input_shapes);
    Graph egraph;
    for (const NodeEntry& n : exp) {
      egraph.outputs.push_back(n);
    }
    egraph = ApplyPasses(egraph, {"InferShape", "ExpandCompute"});
    *ret = egraph.outputs;
    return true;
  };
  return GraphTransform(src, transform);
}

NNVM_REGISTER_PASS(ExpandCompute)
.set_body(ExpandCompute);

}  // namespace compiler
}  // namespace nnvm
