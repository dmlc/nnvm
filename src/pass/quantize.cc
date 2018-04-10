/*!
 *  Copyright (c) 2016 by Contributors
 * \file quantization.cc
 * \brief Quantize the graph to lowbit operator
 */
#include <tvm/runtime/registry.h>
#include <nnvm/graph.h>
#include <nnvm/pass.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/compiler/op_attr_types.h>
#include <nnvm/compiler/packed_func_ext.h>
#include <unordered_set>
#include <string>
#include <cmath>
#include "../compiler/graph_transform.h"

namespace nnvm {
namespace pass {
namespace {

using compiler::FQuantize;

inline NodeEntry MakeQuantizeNode(NodeEntry e, int repr_bit) {
  std::string name = e.node->attrs.name;
  NodeEntry quantize = MakeNode("quantize", name + "_quantize",
    {e}, {{"repr_bit", std::to_string(repr_bit)}, {"out_type", compiler::storage_type}});
  return quantize;
}

inline NodeEntry MakeDequantizeNode(NodeEntry e, int repr_bit) {
  NodeEntry dequantize = MakeNode("dequantize", e.node->attrs.name + "_dequantize",
    {e}, {{"repr_bit", std::to_string(repr_bit)}});
  return dequantize;
}


Graph QuantizeGraph(nnvm::Graph&& src) {
  static auto& fquantize_map = Op::GetAttr<FQuantize>("FQuantize");
  const auto& base2_range = src.GetAttr<std::vector<int>>("base2_range");
  int debug = src.GetAttr<int>("debug");
  const auto& idx = src.indexed_graph();
  std::unordered_map<Node*, NodeEntry> quantized_var;
  std::unordered_map<Node*, int> reverse_mirror;
  // the bit of every quantile
  std::vector<int> repr_bit_map(idx.num_node_entries(), 0);

  std::vector<NodeEntry> debug_outputs;
  auto transform = [&](uint32_t nid, const NodePtr& n, std::vector<NodeEntry>* ret) {
    if (n->is_variable()) return false;
    if (fquantize_map.count(n->op())) {
      NodePtr temp = MakeNode(n->op()->name.c_str(), n->attrs.name, n->inputs, n->attrs.dict).node;
      for (size_t i = 0; i < temp->inputs.size(); ++i) {
        const auto& e = temp->inputs[i];
        if (e.node->is_variable()) {
          if (quantized_var.count(e.node.get())) {
            n->inputs[i] = quantized_var.at(e.node.get());
          } else {
            int k = base2_range[idx.entry_id(e)];
            int repr_bit = k - (compiler::input_bit - 1);
            repr_bit_map[idx.entry_id(e)] = repr_bit;
            NodeEntry quantize = MakeQuantizeNode(e, repr_bit);
            quantized_var.emplace(e.node.get(), quantize);
            temp->inputs[i] = quantize;
          }
        }
      }

      auto fquantize = fquantize_map[n->op()];
      std::vector<int> out_repr_bit(n->num_outputs());
      NodePtr qnode = fquantize(nid, temp, idx, base2_range, repr_bit_map, &out_repr_bit);
      reverse_mirror.emplace(qnode.get(), nid);

      std::vector<NodeEntry> outputs;
      outputs.reserve(qnode->num_outputs());
      for (uint32_t i = 0; i < qnode->num_outputs(); ++i) {
        outputs.emplace_back(NodeEntry{qnode, 0, i});
        repr_bit_map[idx.entry_id(nid, i)] = out_repr_bit[i];
        if (debug) {
          debug_outputs.emplace_back(NodeEntry{qnode, 0, i});
        }
      }
      *ret = std::move(outputs);
      return true;
    } else {
      LOG(FATAL) << n->op()->name << " cannot be quantized yet.";
      return false;
    }
  };

  Graph ret = compiler::GraphTransform(src, transform);
  std::vector<NodeEntry> outputs;
  const std::vector<NodeEntry>& src_outputs = debug ? debug_outputs : ret.outputs;
  outputs.reserve(src_outputs.size());
  for (const auto& e : src_outputs) {
    int eid = idx.entry_id(reverse_mirror.at(e.node.get()), 0);
    NodeEntry dequantize = MakeDequantizeNode(e, repr_bit_map[eid]);
    outputs.emplace_back(dequantize);
  }
  ret.outputs = std::move(outputs);
  return ret;
}

NNVM_REGISTER_PASS(Quantize)
.describe("")
.set_body(QuantizeGraph)
.set_change_graph(true);


Graph CollectInternalOutputs(Graph src, bool include_vars=true) {
  std::vector<NodeEntry> outputs;
  outputs.reserve(src.indexed_graph().num_node_entries());
  DFSVisit(src.outputs, [&](const NodePtr& n) {
      if (!include_vars && n->is_variable()) return;
      for (uint32_t i = 0; i < n->num_outputs(); ++i) {
        outputs.emplace_back(NodeEntry{n, i, 0});
      }
    });

  Graph ret;
  ret.outputs = std::move(outputs);
  return ret;
}

TVM_REGISTER_GLOBAL("nnvm.quantization.CollectInternalOutputs")
.set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue *rv) {
  if (args.size() == 1) {
    *rv = CollectInternalOutputs(args[0]);
  } else {
    *rv = CollectInternalOutputs(args[0], args[1]);
  }
});

}  // namespace
}  // namespace pass
}  // namespace nnvm
