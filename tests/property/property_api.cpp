/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2022 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/type_coverage.h"

#include <string>
#include <type_traits>
#include <utility>

template <typename Property>
void check_property_impl() {
  CHECK(std::is_base_of_v<std::true_type, sycl::is_property<Property>>);
  CHECK(sycl::is_property_v<Property>);
}

template <typename Property>
struct check_property {
  void operator()(const std::string& property_name) {
    INFO("property: " << property_name);
    check_property_impl<Property>();
  }
};

template <typename Property, typename SyclObject>
void check_property_object_impl() {
  CHECK(std::is_base_of_v<std::true_type,
                          sycl::is_property_of<Property, SyclObject>>);
  CHECK(sycl::is_property_of_v<Property, SyclObject>);
}

template <typename Property, typename SyclObject>
struct check_property_object {
  void operator()(const std::string& property_name,
                  const std::string& object_name) {
    INFO("object: " << object_name << ", property: " << property_name);
    check_property_impl<Property>();
    check_property_object_impl<Property, SyclObject>();
  }
};

TEST_CASE("property api", "[property]") {
  {
    const auto properties = named_type_pack<
        sycl::property::queue::enable_profiling,
        sycl::property::queue::in_order>::generate("queue::enable_profiling",
                                                   "queue::in_order");
    const auto objects = named_type_pack<sycl::queue>::generate("queue");
    for_all_combinations<check_property_object>(properties, objects);
  }
  {
    const auto properties =
        named_type_pack<sycl::property::buffer::use_host_ptr,
                        sycl::property::buffer::use_mutex,
                        sycl::property::buffer::context_bound>::
            generate("buffer::use_host_ptr", "buffer::use_mutex",
                     "buffer::context_bound");
    // provide any template argument for the types that require it
    const auto objects = named_type_pack<sycl::buffer<int>>::generate("buffer");
    for_all_combinations<check_property_object>(properties, objects);
  }
  {
    const auto properties = named_type_pack<
        sycl::property::image::use_host_ptr, sycl::property::image::use_mutex,
        sycl::property::image::context_bound>::generate("image::use_host_ptr",
                                                        "image::use_mutex",
                                                        "image::context_bound");
    const auto objects =
        named_type_pack<sycl::sampled_image<>,
                        sycl::unsampled_image<>>::generate("sampled_image",
                                                           "unsampled_image");
    for_all_combinations<check_property_object>(properties, objects);
  }
  {
    const auto properties =
        named_type_pack<sycl::property::no_init>::generate("property::no_init");
    // provide any template argument for the types that require it
    const auto objects = named_type_pack<
        sycl::accessor<int>, sycl::host_accessor<int>,
        sycl::unsampled_image_accessor<sycl::int4, 1, sycl::access_mode::read>,
        sycl::host_unsampled_image_accessor<sycl::int4, 1>>::
        generate("accessor", "host_accessor", "unsampled_image_accessor",
                 "host_unsampled_image_accessor");
    for_all_combinations<check_property_object>(properties, objects);
  }
  {
    const auto properties =
        named_type_pack<sycl::property::reduction::initialize_to_identity>::
            generate("reduction::initialize_to_identity");
    for_all_combinations<check_property>(properties);
  }
}

// Checks that the has_property() member function of the property interface is
// declared noexcept for every SYCL object providing that interface.
namespace property_noexcept {

template <typename Property, typename SyclObject>
constexpr bool check_has_property() {
  CHECK_NOEXCEPT(
      std::declval<const SyclObject&>().template has_property<Property>());
  return true;
}

static_assert(
    check_has_property<sycl::property::queue::enable_profiling, sycl::queue>());
static_assert(
    check_has_property<sycl::property::queue::in_order, sycl::queue>());

static_assert(check_has_property<sycl::property::buffer::use_host_ptr,
                                 sycl::buffer<int>>());
static_assert(
    check_has_property<sycl::property::buffer::use_mutex, sycl::buffer<int>>());
static_assert(check_has_property<sycl::property::buffer::context_bound,
                                 sycl::buffer<int>>());

static_assert(check_has_property<sycl::property::image::use_host_ptr,
                                 sycl::unsampled_image<>>());
static_assert(check_has_property<sycl::property::image::use_mutex,
                                 sycl::unsampled_image<>>());
static_assert(check_has_property<sycl::property::image::context_bound,
                                 sycl::unsampled_image<>>());

static_assert(check_has_property<sycl::property::image::use_host_ptr,
                                 sycl::sampled_image<>>());
static_assert(check_has_property<sycl::property::image::use_mutex,
                                 sycl::sampled_image<>>());
static_assert(check_has_property<sycl::property::image::context_bound,
                                 sycl::sampled_image<>>());

static_assert(
    check_has_property<sycl::property::no_init, sycl::accessor<int>>());
static_assert(
    check_has_property<sycl::property::no_init, sycl::host_accessor<int>>());
static_assert(
    check_has_property<sycl::property::no_init,
                       sycl::unsampled_image_accessor<
                           sycl::int4, 1, sycl::access_mode::read>>());
static_assert(
    check_has_property<sycl::property::no_init,
                       sycl::host_unsampled_image_accessor<sycl::int4, 1>>());

}  // namespace property_noexcept
