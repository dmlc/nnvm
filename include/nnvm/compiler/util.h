/*!
*  Copyright (c) 2016 by Contributors
* \file util.h
* \brief Utility functions for nnvm compiler
*/
#ifndef NNVM_COMPILER_UTIL_H_
#define NNVM_COMPILER_UTIL_H_

#include <tvm/expr.h>
#include <nnvm/tuple.h>

namespace nnvm {
namespace compiler {

/*
 * \brief Helper function to convert TShape to TVM array. Useful for
 * passing data from NNVM param structures to TOPI ops.
 *
 * \param shape The shape to convert
 *
 * \return An Array of Expr, where each element is a constant int32
 */
inline tvm::Array<tvm::Expr> ShapeToArray(TShape shape) {
  tvm::Array<tvm::Expr> result;
  for (auto i : shape) {
    result.push_back(tvm::make_const(tvm::Int(32), i));
  }
  return result;
}

/*
 * \brief Helper function to convert tvm::Tensor shape to TShape. Useful for
 * passing data from NNVM param structures to TOPI ops.
 *
 * \param Tensor from where we need to extract shape
 *
 * \return TShape of the Tensor
 */
inline TShape TensorToShape(tvm::Tensor intensor) {
  TShape result(intensor->shape.size());
  int j = 0;
  for (auto i : intensor->shape) {
    result[j++] = *(as_const_int(i));
  }
  return result;
}

}  // namespace compiler
}  // namespace nnvm
#endif  // NNVM_COMPILER_UTIL_H_
