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
  std::map<std::string, TShape> name2shape;
  const ShapeVector& shape_vec = src.GetAttr<ShapeVector>("shape");
  for (uint32_t i = 0, j = 0; i < idx.num_nodes(); ++i) {
    const Node* src = idx[i].source;
    uint32_t num_outputs = src->num_outputs();
    std::string name = src->attrs.name;
    CHECK(name2shape.count(name) == 0 || name2shape[name] == shape_vec[j])
      << "Reassigning shape of " << name << ". prev: "
      << shape_vec[j] << ", new: " << name2shape[name];
    name2shape[name] = shape_vec[j];
    j += num_outputs;
  }
  bool needs_expand = false;
  auto transform = [&](uint32_t nid, const NodePtr& n, std::vector<NodeEntry>* ret) {
    static auto& fcompute = Op::GetAttr<FExpandCompute>("FExpandCompute");
    if (!fcompute.count(n->op())) return false;
    std::vector<TShape> input_shapes;
    for (const NodeEntry& inp : n->inputs) {
      CHECK_GT(name2shape.count(inp.node->attrs.name), 0)
        << "Input " << inp.node->attrs.name << " as input to "
        << n->attrs.name << " does not exist.";
      input_shapes.push_back(name2shape[inp.node->attrs.name]);
    }
    std::vector<NodeEntry> exp = fcompute[n->op()](n, n->inputs, input_shapes);
    needs_expand = true;
    *ret = exp;
    return true;
  };

  // preserve input shapes
  Graph egraph = GraphTransform(src, transform);
  const IndexedGraph& eidx = egraph.indexed_graph();
  ShapeVector ishapes;
  for (const auto& nid : eidx.input_nodes()) {
    std::string name = eidx[nid].source->attrs.name;
    if (name2shape.count(name)) {
      ishapes.push_back(name2shape[name]);
    } else {
      ishapes.emplace_back();
    }
  }
  egraph.attrs["shape_inputs"] = std::make_shared<any>(std::move(ishapes));

  if (needs_expand)
    return ApplyPasses(egraph, {"InferShape", "ExpandCompute"});
  return egraph;
}

NNVM_REGISTER_PASS(ExpandCompute)
.set_body(ExpandCompute);

}  // namespace compiler
}  // namespace nnvm
