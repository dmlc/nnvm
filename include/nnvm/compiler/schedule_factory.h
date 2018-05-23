/*!
 *  Copyright (c) 2016 by Contributors
 * \file schedule_factory.h
 * \brief Schedule factory information structor.
 */
#ifndef NNVM_COMPILER_SCHEDULE_FACTORY_H_
#define NNVM_COMPILER_SCHEDULE_FACTORY_H_

#include <dmlc/parameter.h>
#include <nnvm/compiler/op_attr_types.h>
#include <string>
#include <vector>
#include <utility>
#include <typeinfo>
#include <limits>
#include <functional>
#include "../c_api.h"
#include "tvm/tvm.h"
#include "tvm/build_module.h"

namespace nnvm {
namespace compiler {

/*! \brief Builder function for instantiating schedules. */
using FTVMScheduleBuilder = std::function<
  tvm::Schedule(const tvm::Target& target, const tvm::Array<tvm::Tensor>& outs)>;

/*!
 * \brief Schedule factory structure
 */
class NNVM_DLL ScheduleFactory {
 public:
  /*! \brief name of the schedule */
  std::string name;
  /* \brief the generic builder */
  FTVMScheduleBuilder generic_builder;
  /* \brief map from keys to registered builders */
  std::unordered_map<std::string, FTVMScheduleBuilder> builders;
  /*!
  * \brief Set the generic schedule for this factory.
  * \param value The builder to instantiate the generic schedule
  * \return reference to self.
  */
  inline ScheduleFactory& set_generic_schedule(const FTVMScheduleBuilder& value);
  /*!
   * \brief Add a specialized schedule
   * \param tags The tags for this specialization
   * \param value The builder to instantiate the specialized schedule
   * \return reference to self.
   */
  inline ScheduleFactory& add_schedule(const std::vector<std::string>& tags,
    const FTVMScheduleBuilder& value);
  /*!
  * \brief Construct a schedule for the given target and output tensors
  * \param target The target to specialize for.
  * \param outs The output tensors
  * \return reference to self.
  */
  tvm::Schedule get_schedule(const tvm::Target& target, const tvm::Array<tvm::Tensor>& outs);
};

/*!
 * \def NNVM_REGISTER_SCHEDULE_FACTORY
 * \brief Register a new schedule factory, or set a device-specific variant
 * of the corresponding schedule.
 *
 * \param name The name of the factory
 */
#define NNVM_REGISTER_SCHEDULE_FACTORY(name)                           \
  DMLC_REGISTRY_REGISTER(::nnvm::compiler::ScheduleFactory, ScheduleFactory, name)

inline ScheduleFactory& ScheduleFactory::set_generic_schedule(const FTVMScheduleBuilder& value) {
  CHECK(generic_builder == nullptr) << "Generic function already registered for " << name;
  this->generic_builder = value;
  return *this;
}

inline ScheduleFactory& ScheduleFactory::add_schedule(const std::vector<std::string>& tags,
                                                      const FTVMScheduleBuilder& value) {
  for (auto &t : tags) {
    auto iter = builders.find(t);
    CHECK(iter == builders.end())
      << "Tag " << t << " already registered for schedule factory " << name;
    builders[t] = value;
  }
  return *this;
}

}  // namespace compiler
}  // namespace nnvm

#endif  // NNVM_COMPILER_SCHEDULE_FACTORY_H_
