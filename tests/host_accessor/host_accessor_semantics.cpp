/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/disabled_for_test_case.h"
#include "../common/semantics_reference.h"
#include <cstddef>
#include <utility>

template <int Dimensions>
struct storage {
  std::size_t byte_size;
  std::size_t size;
  std::size_t max_size;
  bool is_empty;
  sycl::range<Dimensions> range =
      sycl_cts::util::get_cts_object::range<Dimensions>::get(1, 1, 1);
  sycl::id<Dimensions> offset;

  template <typename T, sycl::access_mode AccessMode>
  explicit storage(
      const sycl::host_accessor<T, Dimensions, AccessMode>& host_accessor)
      : byte_size(host_accessor.byte_size()),
        size(host_accessor.size()),
        max_size(host_accessor.max_size()),
        is_empty(host_accessor.empty()),
        range(host_accessor.get_range()),
        offset(host_accessor.get_offset()) {}

  template <typename T, sycl::access_mode AccessMode>
  bool check(const sycl::host_accessor<T, Dimensions, AccessMode>&
                 host_accessor) const {
    return host_accessor.byte_size() == byte_size &&
           host_accessor.size() == size &&
           host_accessor.max_size() == max_size &&
           host_accessor.empty() == is_empty &&
           host_accessor.get_range() == range &&
           host_accessor.get_offset() == offset;
  }
};

TEST_CASE("host_accessor common reference semantics", "[host_accessor]") {
  sycl::buffer<int> buffer_0{sycl::range<1>{1}};
  sycl::host_accessor<int> host_accessor_0{buffer_0};

  sycl::buffer<int> buffer_1{sycl::range<1>{1}};
  sycl::host_accessor<int> host_accessor_1{buffer_1};
  common_reference_semantics::check_host<storage<1>>(
      host_accessor_0, host_accessor_1, "host_accessor");
}

TEST_CASE("host_accessor common reference semantics, mutation",
          "[host_accessor]") {
  constexpr int val = 1;
  constexpr int new_val = 2;
  sycl::buffer<int> buffer{sycl::range<1>{1}};
  sycl::host_accessor<int> t0{buffer};

  SECTION("mutation to copy") {
    t0[0] = val;
    sycl::host_accessor<int> t1(t0);
    t1[0] = new_val;
    CHECK(new_val == t0[0]);
  }

  SECTION("mutation to original") {
    t0[0] = val;
    sycl::host_accessor<int> t1(t0);
    t0[0] = new_val;
    CHECK(new_val == t1[0]);
  }

  SECTION("mutation to original, const copy") {
    t0[0] = val;
    sycl::host_accessor<int> t1(t0);
    t0[0] = new_val;
    CHECK(new_val == t1[0]);
  }
}

// Checks that every member function of sycl::host_accessor that the
// specification declares with a "noexcept" exception specification is actually
// declared that way.
namespace host_accessor_noexcept {

template <int Dimensions, sycl::access_mode AccessMode>
constexpr bool check_host_accessor() {
  using accessor_t = sycl::host_accessor<int, Dimensions, AccessMode>;

  CHECK_NOEXCEPT(std::declval<const accessor_t&>().byte_size());
  CHECK_NOEXCEPT(std::declval<const accessor_t&>().size());
  CHECK_NOEXCEPT(std::declval<const accessor_t&>().empty());
  CHECK_NOEXCEPT(std::declval<const accessor_t&>().get_pointer());
  CHECK_NOEXCEPT(std::declval<const accessor_t&>().begin());
  CHECK_NOEXCEPT(std::declval<const accessor_t&>().end());
  CHECK_NOEXCEPT(std::declval<const accessor_t&>().cbegin());
  CHECK_NOEXCEPT(std::declval<const accessor_t&>().cend());
  CHECK_NOEXCEPT(std::declval<const accessor_t&>().rbegin());
  CHECK_NOEXCEPT(std::declval<const accessor_t&>().rend());
  CHECK_NOEXCEPT(std::declval<const accessor_t&>().crbegin());
  CHECK_NOEXCEPT(std::declval<const accessor_t&>().crend());
#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
  CHECK_NOEXCEPT(std::declval<const accessor_t&>().max_size());
#endif  // SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS

  return true;
}

template <int Dimensions>
constexpr bool check_all() {
  return check_host_accessor<Dimensions, sycl::access_mode::read>() &&
         check_host_accessor<Dimensions, sycl::access_mode::write>() &&
         check_host_accessor<Dimensions, sycl::access_mode::read_write>();
}

static_assert(check_all<1>());
static_assert(check_all<2>());
static_assert(check_all<3>());

}  // namespace host_accessor_noexcept
