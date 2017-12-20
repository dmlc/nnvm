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
}·

int NNBackwardGraphCreate(SymbolHandle symbol,
		                  const std::vector<OpReqType>& grad_req_types,
						  const std::vector<SymbolHandle>& head_grads,
						  GraphHandle *graph) {
  Graph* g = new Graph();
  API_BEGIN();
  g->outputs = static_cast<Symbol*>(symbol)->outputs;
  bool need_grad = false;
  for (OpReqType req : grad_req_types) {
      if (req != kNullOp) need_grad = true;
  }
  CHECK(need_grad)
    << "Trying to build backward pass on a graph which doesn't require gradient. "
	<< "At least one node with grad_req rather than 'null' is required in the "
	<< "graph to call backward pass creation function.";

  // Setup head grad entry
  bool has_head_grad = false;
  if (head_grads.size() > 0) {
	  CHECK_EQ(g->outputs.size(), head_grads.size())
	    << "Graph output number and head gradient number mismatch.";
	  has_head_grad =true;
  }
  std::vector<NodeEntry> head_grad_entry;
  head_grad_entry.resize(g->outputs.size());
  for (size_t i = 0; i < g->outputs.size(); ++i) {
    if (has_head_grad) {
      auto& head_symbol_outputs = static_cast<Symbol*>(head_grads[i])->outputs;
      CHECK_EQ(head_symbol_outputs, 1)
        << "Each head grad symbol must contain only one output.";
    	  auto& head_grad_node = head_symbol_outputs.front();
    	  CHECK(head_grad_node.node->is_variable())
    	    << "Each head grad symbol must be a placeholder variable.";
    	  head_grad_entry.emplace_back(NodeEntry{head_grad_node, 0, 0});
    }
    else {
    	  NodePtr one_node = Node::Create();
      one_node->attrs.op = Op::Get("__one__");
    	  head_grad_entry.emplace_back(NodeEntry{one_node, 0, 0});
    }
  }

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
