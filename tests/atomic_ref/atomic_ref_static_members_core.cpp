/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides sycl::atomic_ref static members test for generic types
//
*******************************************************************************/

#include "catch2/catch_test_macros.hpp"

#include "atomic_ref_static_members.h"
#include <utility>

namespace atomic_ref::tests::static_members::core {

TEST_CASE("sycl::atomic_ref static members. core types", "[atomic_ref]") {
  const auto types = atomic_ref::tests::common::get_conformance_type_pack();
  for_all_types<atomic_ref::tests::static_members::run_test>(types);
}

}  // namespace atomic_ref::tests::static_members::core

// Checks that every member function of sycl::atomic_ref that the specification
// declares with a "noexcept" exception specification is actually declared that
// way.
namespace atomic_ref_noexcept {

constexpr auto default_order = sycl::memory_order::relaxed;
constexpr auto default_scope = sycl::memory_scope::device;

/** Members that are common to every sycl::atomic_ref specialization. */
template <typename T, sycl::access::address_space AddressSpace>
constexpr bool check_common_members() {
  using atomic_ref_t =
      sycl::atomic_ref<T, default_order, default_scope, AddressSpace>;

  CHECK_NOEXCEPT(std::declval<const atomic_ref_t&>().is_lock_free());
  CHECK_NOEXCEPT(atomic_ref_t(std::declval<const atomic_ref_t&>()));
  CHECK_NOEXCEPT(std::declval<const atomic_ref_t&>().store(std::declval<T>()));
  CHECK_NOEXCEPT(std::declval<const atomic_ref_t&>() = std::declval<T>());
  CHECK_NOEXCEPT(std::declval<const atomic_ref_t&>().load());
  CHECK_NOEXCEPT(static_cast<T>(std::declval<const atomic_ref_t&>()));
  CHECK_NOEXCEPT(
      std::declval<const atomic_ref_t&>().exchange(std::declval<T>()));
  CHECK_NOEXCEPT(std::declval<const atomic_ref_t&>().compare_exchange_weak(
      std::declval<T&>(), std::declval<T>(), default_order, default_order));
  CHECK_NOEXCEPT(std::declval<const atomic_ref_t&>().compare_exchange_weak(
      std::declval<T&>(), std::declval<T>()));
  CHECK_NOEXCEPT(std::declval<const atomic_ref_t&>().compare_exchange_strong(
      std::declval<T&>(), std::declval<T>(), default_order, default_order));
  CHECK_NOEXCEPT(std::declval<const atomic_ref_t&>().compare_exchange_strong(
      std::declval<T&>(), std::declval<T>()));

  return true;
}

/** Members of the sycl::atomic_ref specialization for integral types. */
template <typename T, sycl::access::address_space AddressSpace>
constexpr bool check_integral_members() {
  using atomic_ref_t =
      sycl::atomic_ref<T, default_order, default_scope, AddressSpace>;

  CHECK_NOEXCEPT(
      std::declval<const atomic_ref_t&>().fetch_add(std::declval<T>()));
  CHECK_NOEXCEPT(
      std::declval<const atomic_ref_t&>().fetch_sub(std::declval<T>()));
  CHECK_NOEXCEPT(
      std::declval<const atomic_ref_t&>().fetch_and(std::declval<T>()));
  CHECK_NOEXCEPT(
      std::declval<const atomic_ref_t&>().fetch_or(std::declval<T>()));
  CHECK_NOEXCEPT(
      std::declval<const atomic_ref_t&>().fetch_xor(std::declval<T>()));
  CHECK_NOEXCEPT(
      std::declval<const atomic_ref_t&>().fetch_min(std::declval<T>()));
  CHECK_NOEXCEPT(
      std::declval<const atomic_ref_t&>().fetch_max(std::declval<T>()));
  CHECK_NOEXCEPT(std::declval<const atomic_ref_t&>()++);
  CHECK_NOEXCEPT(std::declval<const atomic_ref_t&>()--);
  CHECK_NOEXCEPT(++std::declval<const atomic_ref_t&>());
  CHECK_NOEXCEPT(--std::declval<const atomic_ref_t&>());
  CHECK_NOEXCEPT(std::declval<const atomic_ref_t&>() += std::declval<T>());
  CHECK_NOEXCEPT(std::declval<const atomic_ref_t&>() -= std::declval<T>());
  CHECK_NOEXCEPT(std::declval<const atomic_ref_t&>() &= std::declval<T>());
  CHECK_NOEXCEPT(std::declval<const atomic_ref_t&>() |= std::declval<T>());
  CHECK_NOEXCEPT(std::declval<const atomic_ref_t&>() ^= std::declval<T>());

  return true;
}

/** Members of the sycl::atomic_ref specialization for floating-point types. */
template <typename T, sycl::access::address_space AddressSpace>
constexpr bool check_floating_point_members() {
  using atomic_ref_t =
      sycl::atomic_ref<T, default_order, default_scope, AddressSpace>;

  CHECK_NOEXCEPT(
      std::declval<const atomic_ref_t&>().fetch_add(std::declval<T>()));
  CHECK_NOEXCEPT(
      std::declval<const atomic_ref_t&>().fetch_sub(std::declval<T>()));
  CHECK_NOEXCEPT(
      std::declval<const atomic_ref_t&>().fetch_min(std::declval<T>()));
  CHECK_NOEXCEPT(
      std::declval<const atomic_ref_t&>().fetch_max(std::declval<T>()));
  CHECK_NOEXCEPT(std::declval<const atomic_ref_t&>() += std::declval<T>());
  CHECK_NOEXCEPT(std::declval<const atomic_ref_t&>() -= std::declval<T>());

  return true;
}

/** Members of the sycl::atomic_ref specialization for pointer types. */
template <typename T, sycl::access::address_space AddressSpace>
constexpr bool check_pointer_members() {
  using atomic_ref_t =
      sycl::atomic_ref<T*, default_order, default_scope, AddressSpace>;
  using difference_type = typename atomic_ref_t::difference_type;

  CHECK_NOEXCEPT(std::declval<const atomic_ref_t&>().fetch_add(
      std::declval<difference_type>()));
  CHECK_NOEXCEPT(std::declval<const atomic_ref_t&>().fetch_sub(
      std::declval<difference_type>()));
  CHECK_NOEXCEPT(std::declval<const atomic_ref_t&>()++);
  CHECK_NOEXCEPT(std::declval<const atomic_ref_t&>()--);
  CHECK_NOEXCEPT(++std::declval<const atomic_ref_t&>());
  CHECK_NOEXCEPT(--std::declval<const atomic_ref_t&>());
  CHECK_NOEXCEPT(std::declval<const atomic_ref_t&>() +=
                 std::declval<difference_type>());
  CHECK_NOEXCEPT(std::declval<const atomic_ref_t&>() -=
                 std::declval<difference_type>());

  return true;
}

template <sycl::access::address_space AddressSpace>
constexpr bool check_address_space() {
  return check_common_members<int, AddressSpace>() &&
         check_integral_members<int, AddressSpace>() &&
         check_common_members<float, AddressSpace>() &&
         check_floating_point_members<float, AddressSpace>() &&
         check_common_members<int*, AddressSpace>() &&
         check_pointer_members<int, AddressSpace>();
}

static_assert(
    check_address_space<sycl::access::address_space::generic_space>());
static_assert(check_address_space<sycl::access::address_space::global_space>());
static_assert(check_address_space<sycl::access::address_space::local_space>());

}  // namespace atomic_ref_noexcept
