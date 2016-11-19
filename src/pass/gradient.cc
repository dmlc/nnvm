/*!
 *  Copyright (c) 2016 by Contributors
 * \file gradients.cc
 * \brief Passes that takes gradient of the graph
 * This code code was modified based on mxnet codebase by Min Lin
 */
#include <nnvm/pass.h>
#include <nnvm/pass_functions.h>
#include <nnvm/op_attr_types.h>
#include <algorithm>
#include <functional>

using namespace std;

namespace nnvm {
namespace pass {
namespace {

// default aggregate gradient function
// require operator __zero__ and __ewise_sum__ to be presented.
NodeEntry DefaultAggregateGradient(vector<NodeEntry>&& v) {
  if (v.size() == 1) {
    return std::move(v[0]);
  } else if (v.size() == 0) {
    NodePtr zero_node = Node::Create();
    zero_node->attrs.op = Op::Get("__zero__");
    return NodeEntry{zero_node, 0, 0};
  } else {
    NodePtr sum_node = Node::Create();
    sum_node->attrs.op = Op::Get("__ewise_sum__");
    sum_node->inputs = std::move(v);
    return NodeEntry{sum_node, 0, 0};
  }
}

// helper entry
struct GradEntry {
#ifdef _MSC_VER
  NodeEntry sum = NodeEntry{nullptr, 0, 0};
#else
  NodeEntry sum{nullptr, 0, 0};
#endif
  vector<NodeEntry> grads;
};

class GradientPass : public Pass {
 public:
  PassResult RunOnGraph(Graph src, const PassArgument& pargs) override {
    using nnvm::FGradient;
    using MirrorFun = function<int (const Node& node)>;
    using AggFun = function<NodeEntry (vector<NodeEntry>&& inputs)>;

    const GradientPassArgs& args = nnvm::get<GradientPassArgs>(pargs.value);

    AggFun agg_fun = args.aggregate_fun?
      args.aggregate_fun : DefaultAggregateGradient;

    // topo sort
    vector<NodePtr> topo_order;
    unordered_map<Node*, vector<GradEntry> > output_grads;

    DFSVisit(args.ys, [&](const NodePtr& node) {
        if (output_grads.count(node.get()) == 0) {
          output_grads[node.get()].resize(node->num_outputs());
        }
        topo_order.push_back(node);
      });

    CHECK_EQ(args.ys.size(), args.ys_out_grad.size());
    for (size_t i = 0; i < args.ys.size(); ++i) {
      output_grads[args.ys[i].node.get()][args.ys[i].index].grads = { args.ys_out_grad[i] };
    }

    // construct mirror reduece memory strategy if needed
    unordered_map<Node*, NodePtr> mirror_map;
    if (!args.mirror_fun) {
      for (const NodePtr& n : topo_order) {
        if (args.mirror_fun(*n)) {
          NodePtr new_node = Node::Create();
          *new_node = *n;
          new_node->attrs.name += "_mirror";
          for (auto& e : new_node->inputs) {
            e.node = mirror_map.at(e.node.get());
          }
          for (auto& n : new_node->control_deps) {
            n = mirror_map.at(n.get());
          }
          mirror_map[n.get()] = std::move(new_node);
        } else {
          mirror_map[n.get()] = n;
        }
      }
    }

    // traverse backward
    static auto& grad_fun_map = Op::GetAttr<FGradient>("FGradient");
    vector<NodeEntry> out_agg_grads;
    for (auto rit = topo_order.rbegin(); rit != topo_order.rend(); ++rit) {
      const NodePtr& ptr = *rit;
      if (ptr->is_variable()) continue;
      out_agg_grads.clear();
      for (GradEntry& e : output_grads.at(ptr.get())) {
        e.sum = agg_fun(std::move(e.grads));
        out_agg_grads.push_back(e.sum);
      }
      if ((*rit)->inputs.size() != 0) {
        vector<NodeEntry> input_grads = grad_fun_map[ptr->op()]
            (mirror_map.size() == 0 ? ptr : mirror_map.at(ptr.get()), out_agg_grads);
        CHECK_EQ((*rit)->inputs.size(), input_grads.size())
            << "Gradient function not returning enough gradient";
        auto git = input_grads.begin();
        for (auto it = (*rit)->inputs.begin(); it != (*rit)->inputs.end(); ++it, ++git) {
          output_grads[it->node.get()][it->index].grads.emplace_back(std::move(*git));
        }
      }
    }
    // take out the xs' grads
    PassResult ret;
    ret.graph.outputs.reserve(args.xs.size());
    for (const NodeEntry& e : args.xs) {
      GradEntry& entry = output_grads[e.node.get()][e.index];
      // aggregate sum if there haven't been
      if (entry.sum.node.get() == nullptr) {
        entry.sum = agg_fun(std::move(entry.grads));
      }
      ret.graph.outputs.emplace_back(std::move(entry.sum));
    }
    // TODO(minjie): remap forward node in the backward graph.
    return ret;
  }
};

// register pass
NNVM_REGISTER_PASS_CLASS(GradientPass)
.describe("Return a gradient graph of src.attrs[\"ys\"] wrt src.attrs[\"xs\"]")
.set_change_graph(true);

}  // namespace
}  // namespace pass
}  // namespace nnvm
