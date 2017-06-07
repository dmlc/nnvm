#ifndef NNVM_CODEGEN_H_
#define NNVM_CODEGEN_H_

#include <memory>
#include <vector>
#include "./tuple.h"
#include "./node.h"

namespace nnvm {

class CodeGen {
 public:
  virtual void FeedInputShapes(const std::vector<TShape>& shapes) = 0;
  virtual void FeedOutputShapes(const std::vector<TShape>& shapes) = 0;
  virtual void FeedInputTypes(const std::vector<int>& types) = 0;
  virtual void Generate(const Node* node) = 0;
};

}  // namespace nnvm

#endif  // NNVM_CODEGEN_H_
