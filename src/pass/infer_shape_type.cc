/*!
 *  Copyright (c) 2016 by Contributors
 * \file infer_shape.cc
 * \brief Inference the shapes given existin information.
 */
#include <nnvm/pass.h>
#include <nnvm/pass_functions.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/graph_attr_types.h>

using namespace std;

namespace nnvm {
namespace pass {
namespace {

// Inference attributes of each node/entry using hints from users. Hints could be passed
// in following ways in descending priority:
// 1. Directly specify the attribute in node's attribute dictionary.
// 2. Attributes given for the input nodes.
// 3. Inference function registered for each operator.
// 4. Default inference function.
// The result will be saved under name specifiec by `attr_name`. It will also output extra
// information like how many nodes have unknown attributes after the full inference.
// Example:
//   Shape inference will produce a NodeEntry attribute "shape", and a Graph attribute
//   "shape_num_unknown_nodes".
//
// Example:
//   Type inference will produce a NodeEntry attribute "dtype", and a Graph attribute
//   "dtype_num_unknown_nodes".
template<typename AttrType, typename FIsNone, typename FDefault>
class InferAttrPass : public Pass {
 public:
  InferAttrPass(const AttrType& empty_val,
                const string& infer_functor_name,
                const string& attr_name,
                FIsNone is_none_functor,
                FDefault default_infer_functor):
    empty_val_(empty_val), infer_functor_name_(infer_functor_name),
    attr_name_(attr_name), is_none_functor_(is_none_functor),
    default_infer_functor_(default_infer_functor),
    finfer_shape_(Op::GetAttr<FInferNodeEntryAttr<AttrType> >(infer_functor_name_)),
    backward_map_(Op::GetAttr<FBackwardOutToInIndex>("FBackwardOutToInIndex")),
    backward_in_grad_(Op::GetAttr<FBackwardInGradIndex>("FBackwardInGradIndex")) {
  }

  PassResult RunOnGraph(Graph src, const PassArgument& pargs) {
    using AttrVector = vector<AttrType>;
    const IndexedGraph& idx = src.indexed_graph();
    const InferAttrPassArgs<AttrType>& args =
      nnvm::get<InferAttrPassArgs<AttrType>>(*pargs.value);

    CHECK(!src.HasAttr(attr_name_));
    // reshape shape vector
    AttrVector ret_attrs(idx.num_node_entries(), empty_val_);

    if (!args.input_attrs.empty()) {
      CHECK_LE(args.input_attrs.size(), idx.input_nodes().size())
          << "More provided shapes than number of arguments.";
      for (size_t i = 0; i < args.input_attrs.size(); ++i) {
        ret_attrs[idx.entry_id(idx.input_nodes()[i], 0)] = args.input_attrs[i];
      }
    }
    
    size_t num_unknown = 0;
    const int kMaxStep = 3;
    for (int i = 0; i < kMaxStep; ++i) {
      if (i % 2 == 0) {
        for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
          InferOneNode(nid, idx, args, &ret_attrs);
        }
      } else {
        // backward inference
        for (uint32_t i = idx.num_nodes(); i != 0; --i) {
          InferOneNode(i - 1, idx, args, &ret_attrs);
        }
      }
      num_unknown = 0;
      for (size_t i = 0; i < idx.num_node_entries(); ++i) {
        if (fis_none_(ret_attrs[i])) {
          ++num_unknown;
        }
      }
      if (num_unknown == 0) break;
    }
    PassResult ret;
    ret.graph = src;
    // set the shapes
    ret.graph.SetNodeEntryAttr(attr_name_, std::make_shared<any>(std::move(ret_attrs)));
    // number of nodes who knows the shape.
    ret.graph.SetGraphAttr(attr_name_ + "_num_unknown_nodes",
                           std::make_shared<any>(num_unknown));
    return ret;
  }

 private:
  // Inference step function for one node.
  void InferOneNode(uint32_t nid, const IndexedGraph& idx,
                    const InferAttrPassArgs<AttrType>& args,
                    vector<AttrType>* ret_attrs) {
    // Temp space for shape inference.
    vector<AttrType> ishape, oshape;

    const auto& inode = idx[nid];
    const uint32_t num_inputs = inode.inputs.size();
    const uint32_t num_outputs = inode.source->num_outputs();

    if (inode.source->is_variable()) {
      // Variable node. No operator. Only one output entry.
      CHECK(!inode.source->op());
      CHECK_EQ(num_outputs, 1);
      const uint32_t out_ent_id = idx.entry_id(nid, 0);
      if (!args.node_attr_key.empty() &&
          is_none_functor_((*ret_attrs)[out_ent_id])) {
        auto it = inode.source->attrs.dict.find(args.node_attr_key);
        if (it != inode.source->attrs.dict.end()) {
          istringstream is(it->second);
          CHECK(is >> (*ret_attrs)[out_ent_id]) << "Invalid attribute";
        }
      }
    } else if (backward_map_.count(inode.source->op())) {
      // Backward operator inference.
      CHECK_GE(inode.control_deps.size(), 1)
        << "BackwardOp need to have control_deps to its forward op";
      const IndexedGraph::Node& fnode = idx[inode.control_deps[0]];
      // Inference the outputs of backward operator (equal to the inputs
      // of its corresponding forward operator).
      vector<uint32_t> out_map =
          backward_map_[inode.source->op()](inode.source->attrs);
      for (size_t i = 0; i < out_map.size(); ++i) {
        uint32_t in_id = out_map[i];
        CHECK_LT(in_id, fnode.inputs.size());
        (*ret_attrs)[idx.entry_id(nid, i)] =
            (*ret_attrs)[idx.entry_id(fnode.inputs[in_id])];
      }
      if (backward_in_grad_.count(inode.source->op())) {
        vector<uint32_t> in_grad =
          backward_in_grad_[inode.source->op()](inode.source->attrs);
        CHECK_LE(in_grad.size(), fnode.source->num_outputs());
        for (size_t i = 0; i < in_grad.size(); ++i) {
          uint32_t eid = idx.entry_id(inode.inputs[in_grad[i]]);
          if (fis_none((*ret_attrs)[eid])) {
            (*ret_attrs)[eid] = (*ret_attrs)[idx.entry_id(inode.control_deps[0], i)];
          }
        }
      }
    } else {
      bool forward_known = true;
      // Forward operator inference.
      ishape.resize(num_inputs, empty_val_);
      for (uint32_t i = 0; i < ishape.size(); ++i) {
        ishape[i] = (*ret_attrs)[idx.entry_id(inode.inputs[i])];
        if (fis_none(ishape[i])) forward_known = false;
      }
      oshape.resize(num_outputs, empty_val_);
      for (uint32_t i = 0; i < oshape.size(); ++i) {
        oshape[i] = (*ret_attrs)[idx.entry_id(nid, i)];
        if (fis_none(oshape[i])) forward_known = false;
      }
      if (!forward_known) {
        auto finfer = finfer_shape_.get(inode.source->op(), default_infer_functor_);
        CHECK(finfer != nullptr)
          << "Attribute " << infer_functor_name_
          << " is not registed by op " << inode.source->op()->name;
        // Call inference function of the operator.
        try {
          forward_known = finfer(inode.source->attrs, &ishape, &oshape);
        } catch (const exception& e) {
          throw dmlc::Error(e.what() + string(" with ") + inode.source->attrs.name);
        }
      }
      // Save to the result map.
      for (uint32_t i = 0; i < num_inputs; ++i) {
        (*ret_attrs)[idx.entry_id(inode.inputs[i])] = ishape[i];
      }
      for (uint32_t i = 0; i < num_outputs; ++i) {
        (*ret_attrs)[idx.entry_id(nid, i)] = oshape[i];
      }
    }
  }

  const AttrType empty_val_;
  const string infer_functor_name_;
  const string attr_name_;
  const FIsNone is_none_functor_;
  const FDefault default_infer_functor_;
  const OpMap<FInferNodeEntryAttr<AttrType>>& finfer_shape_;
  const OpMap<FBackwardOutToInIndex>& backward_map_;
  const OpMap<FBackwardInGradIndex>& backward_in_grad_;
};

template<typename AttrType, typename FIsNone, typename FDefault>
unique_ptr<Pass> CreateInferAttrPass(
    const AttrType& empty_val,
    const string& infer_functor_name,
    const string& attr_name,
    FIsNone is_none_functor,
    FDefault default_infer_functor) {
  return unique_ptr<Pass>(
      new InferAttrPass<AttrType, FIsNone, FDefault>(
        empty_val, infer_functor_name, attr_name,
        is_none_functor, default_infer_functor));
}

NNVM_REGISTER_PASS(InferShape)
.describe("Infer the shape of each node entries.")
.set_body([]() {
    return CreateInferAttrPass(
        TShape(), "FInferShape", "shape",
        [](const TShape& s) { return s.ndim() == 0; },
        nullptr);
    })
.set_change_graph(false)
.provide_graph_attr("shape");

// Inference function for operators that have same types for inputs/outputs.
// The function will try find known types in the given inputs and outputs,
// and then set the remaining types to be the same. Returns whether the type
// is known for inputs and outputs.
static const int kUnknownType = -1;
inline bool SameType(const NodeAttrs& ,
                     vector<int> *iattr,
                     vector<int> *oattr) {
  int def_v = kUnknownType;
  // First search in the inputs.
  auto iter = std::find_if(iattr->begin(), iattr->end(),
                           [](int v) { return v != kUnknownType; });
  if (iter == iattr->end()) {
    // Then search in the outputs.
    iter = std::find_if(oattr->begin(), oattr->end(),
                        [](int v) { return v != kUnknownType; });
    if (iter != oattr->end()) {
      def_v = *iter;
    }
  } else {
    def_v = *iter;
  }
  if (def_v == kUnknownType) {
    return false;
  } else {
    std::fill(iattr->begin(), iattr->end(), def_v);
    std::fill(oattr->begin(), oattr->end(), def_v);
    return true;
  }
}

NNVM_REGISTER_PASS(InferType)
.describe("Infer the dtype of each node entries.")
.set_body([]() {
    return CreateInferAttrPass(
        -1, "FInferType", "dtype",
        [](const int t) { return t == -1; },
        SameType);
  })
.set_change_graph(false)
.provide_graph_attr("dtype");

DMLC_JSON_ENABLE_ANY(ShapeVector, list_shape);
DMLC_JSON_ENABLE_ANY(DTypeVector, list_int);
DMLC_JSON_ENABLE_ANY(size_t, size_t);

}  // namespace
}  // namespace pass
}  // namespace nnvm
