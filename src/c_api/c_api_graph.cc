/*!
 *  Copyright (c) 2016 by Contributors
 * \file c_api_graph.cc
 * \brief C API related to Graph IR.
 */
#include <nnvm/c_api.h>
#include <nnvm/op.h>
#include <nnvm/symbolic.h>
#include <nnvm/graph.h>
#include <nnvm/pass.h>
#include <nnvm/pass_functions.h>
#include <dmlc/json.h>
#include <vector>
#include "./c_api_common.h"

using namespace nnvm;

int NNGraphCreate(SymbolHandle symbol, GraphHandle *graph) {
  Graph* g = new Graph();
  API_BEGIN();
  g->outputs = static_cast<Symbol*>(symbol)->outputs;
  *graph = g;
  API_END_HANDLE_ERROR(delete g);
}

int NNFullGraphCreate(SymbolHandle symbol,
                      const char** fixed_args,
                      const unsigned int num_fixed_args,
                      SymbolHandle* head_grads,
                      const unsigned int num_head_grads,
                      GraphHandle *graph) {
  Graph* g = new Graph();
  API_BEGIN();
  auto* symbol_ptr = static_cast<Symbol*>(symbol);
  g->outputs = symbol_ptr->outputs;

  // Setup backward output nodes
  std::unordered_map<std::string, NodePtr> arg_name_map;
  const auto arg_names = symbol_ptr->ListInputNames(Symbol::kReadOnlyArgs);
  const auto args = symbol_ptr->ListInputs(Symbol::kReadOnlyArgs);
  CHECK_EQ(arg_names.size(), args.size());
  if (arg_names.size() == num_fixed_args) {
    *graph = g;
    return 0;
  }
  for (size_t i = 0; i < arg_names.size(); ++i) {
    arg_name_map[arg_names[i]] = args[i];
  }
  for (size_t i = 0; i < num_fixed_args; ++i) {
    arg_name_map.erase(std::string(fixed_args[i]));
  }
  std::vector<NodeEntry> xs;
  for (auto kv : arg_name_map) {
    xs.emplace_back(NodeEntry{kv.second, 0, 0});
  }

  // Setup head grad entry
  bool has_head_grad = false;
  if (num_head_grads > 0) {
    CHECK_EQ(g->outputs.size(), num_head_grads)
      << "Graph output number and head gradient number mismatch.";
    has_head_grad = true;
  }
  std::vector<NodeEntry> head_grad_entry;
  for (size_t i = 0; i < g->outputs.size(); ++i) {
    if (has_head_grad) {
      auto& head_symbol_outputs = static_cast<Symbol*>(head_grads[i])->outputs;
      CHECK_EQ(head_symbol_outputs.size(), 1)
        << "Each head grad symbol must contain only one output.";
      auto& head_grad_node = head_symbol_outputs.front();
      CHECK(head_grad_node.node->is_variable())
        << "Each head grad symbol must be a placeholder variable.";
      head_grad_entry.push_back(std::move(head_grad_node));
    } else {
      NodePtr one_node = Node::Create();
      one_node->attrs.op = Op::Get("ones_like");
      one_node->inputs.push_back(g->outputs[i]);
      head_grad_entry.emplace_back(NodeEntry{one_node, 0, 0});
    }
  }

  // Create backward graph
  // TODO(yaow): Add mirror function to optimize memory?
  std::vector<const Op*> zero_ops;
  zero_ops.push_back(Op::Get("zeros_like"));
  zero_ops.push_back(Op::Get("_zeros"));
  Graph g_grad = pass::Gradient(
    *g, symbol_ptr->outputs, xs, head_grad_entry,
    nullptr, nullptr, nullptr, zero_ops, "copy");
    CHECK_EQ(g_grad.outputs.size(), xs.size());
  for (const auto &e : g_grad.outputs) {
    g->outputs.push_back(e);
  }

  *graph = g;
  API_END_HANDLE_ERROR(delete g);
}

int NNGraphFree(GraphHandle handle) {
  API_BEGIN();
  delete static_cast<Graph*>(handle);
  API_END();
}

int NNGraphGetSymbol(GraphHandle graph, SymbolHandle *symbol) {
  Symbol* s = new Symbol();
  API_BEGIN();
  s->outputs = static_cast<Graph*>(graph)->outputs;
  *symbol = s;
  API_END_HANDLE_ERROR(delete s);
}

int NNGraphSetNodeEntryListAttr_(GraphHandle handle,
                                 const char* key,
                                 SymbolHandle list) {
  API_BEGIN();
  Symbol* s = static_cast<Symbol*>(list);
  Graph* g = static_cast<Graph*>(handle);
  g->attrs[std::string(key)]
      = std::make_shared<any>(s->outputs);
  API_END();
}

int NNGraphSetJSONAttr(GraphHandle handle,
                       const char* key,
                       const char* json_value) {
  API_BEGIN();
  Graph* g = static_cast<Graph*>(handle);
  std::string temp(json_value);
  std::istringstream is(temp);
  dmlc::JSONReader reader(&is);
  nnvm::any value;
  reader.Read(&value);
  g->attrs[std::string(key)] = std::make_shared<any>(std::move(value));
  API_END();
}

int NNGraphGetJSONAttr(GraphHandle handle,
                      const char* key,
                      const char** json_out,
                      int *success) {
  NNAPIThreadLocalEntry *ret = NNAPIThreadLocalStore::Get();
  API_BEGIN();
  Graph* g = static_cast<Graph*>(handle);
  std::string skey(key);
  auto it = g->attrs.find(skey);
  if (it != g->attrs.end()) {
    std::ostringstream os;
    dmlc::JSONWriter writer(&os);
    writer.Write(*it->second.get());
    ret->ret_str = os.str();
    *json_out = (ret->ret_str).c_str();
    *success = 1;
  } else {
    *success = 0;
  }
  API_END();
}

int NNGraphApplyPasses(GraphHandle src,
                       nn_uint num_pass,
                       const char** pass_names,
                       GraphHandle *dst) {
  Graph* g = new Graph();
  API_BEGIN();
  std::vector<std::string> vpass;
  for (nn_uint i = 0; i < num_pass; ++i) {
    vpass.emplace_back(std::string(pass_names[i]));
  }
  *g = ApplyPasses(*static_cast<Graph*>(src), vpass);
  *dst = g;
  API_END_HANDLE_ERROR(delete g);
}
