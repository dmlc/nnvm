#ifndef NNVM_FRAME_H_
#define NNVM_FRAME_H_

#include <vector>
#include <unordered_map>
#include "./base.h"

namespace nnvm {

class ColumnRef;
struct Column {
  any value;
  std::vector<ColumnRef> sub_columns_;
};
/*!
 * \brief A copy on write object.
 */
class ColumnRef {
 public:
  Column& Get();
  const Column& Get() const;
 private:
  std::shared_ptr<Column> ref_;
};

class Frame {
 public:
  template<typename T>
  void AddAttr(const std::string& attr_name, const T& val);

  template<typename T>
  const T& GetAttr(const std::string& attr_name) const;

  template<typename T>
  T& GetAttr(const std::string& attr_name);

  size_t NumAttributes() const;

  bool HasAttribute() const;

 protected:
  std::unordered_map<std::string, ColumnRef> columns_;
};

/*!
 * \brief A column storage for attributes.
 */
class ColumnFrame {
 public:
  template<typename T>
  void AddAttr(const std::string& attr_name, const std::vector<T>& val);

  template<typename T>
  const std::vector<T>& GetAttr(const std::string& attr_name) const;

  template<typename T>
  std::vector<T>& GetAttr(const std::string& attr_name);

  size_t NumAttributes() const;

  bool HasAttribute() const;

  size_t ColumnLength() const;

 private:
  // Each value in the map is a vector. All vectors should have same length.
  std::unordered_map<std::string, ColumnRef> columns_;
  size_t length_ = 0;
};

}  // namespace nnvm

#endif  // NNVM_FRAME_H_
