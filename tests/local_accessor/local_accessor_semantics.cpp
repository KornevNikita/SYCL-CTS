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
#ifndef SYCL_CTS_COMPILING_WITH_ADAPTIVECPP
#include "../common/semantics_reference.h"
#include <cstddef>
#include <utility>
#endif

template <int Dimensions>
struct storage {
  std::size_t byte_size;
  std::size_t size;
  std::size_t max_size;
  bool is_empty;

  template <typename T>
  explicit storage(const sycl::local_accessor<T, Dimensions>& local_accessor)
      : byte_size(local_accessor.byte_size()),
        size(local_accessor.size()),
        max_size(local_accessor.max_size()),
        is_empty(local_accessor.empty()) {}

  template <typename T>
  bool check(const sycl::local_accessor<T, Dimensions>& local_accessor) const {
    return local_accessor.byte_size() == byte_size &&
           local_accessor.size() == size &&
           local_accessor.max_size() == max_size &&
           local_accessor.empty() == is_empty;
  }
};

DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("local_accessor common reference semantics (host)", "[local_accessor]")({
  sycl_cts::util::get_cts_object::queue().submit([&](sycl::handler& cgh) {
    sycl::local_accessor<int, 0> local_accessor_0{cgh};
    sycl::local_accessor<int, 0> local_accessor_1{cgh};
    common_reference_semantics::check_host<storage<0>>(
        local_accessor_0, local_accessor_1, "local_accessor<int, 0>");
  });
});

DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("local_accessor common reference semantics (kernel)", "[local_accessor]")({
  using type = sycl::local_accessor<int, 0>;
  common_reference_semantics::check_kernel<storage<0>, type>(
      [&](sycl::handler& cgh) { return sycl::local_accessor<int, 0>{cgh}; },
      "local_accessor<int, 0>");
});

template <int TestCase>
struct kernel_name_local;

DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("local_accessor common reference semantics, mutation (kernel)",
 "[local_accessor]")({
  sycl::queue queue = sycl_cts::util::get_cts_object::queue();
  int result = 0;

  constexpr int val = 1;
  constexpr int new_val = 2;

  SECTION("mutation to copy") {
    {
      sycl::buffer<int> buffer_result{&result, sycl::range<1>{1}};
      queue.submit([&](sycl::handler& cgh) {
        auto acc_result = buffer_result.get_access(cgh);
        sycl::local_accessor<int> t0{sycl::range<1>{1}, cgh};
        cgh.parallel_for<kernel_name_local<0>>(
            sycl::nd_range<1>{sycl::range<1>{1}, sycl::range<1>{1}},
            [=](sycl::nd_item<1> nd_item) {
              t0[0] = val;
              sycl::local_accessor<int> t1(t0);
              t1[0] = new_val;
              acc_result[0] = t0[0];
            });
      });
    }
    CHECK(new_val == result);
  }

  SECTION("mutation to original") {
    {
      sycl::buffer<int> buffer_result{&result, sycl::range<1>{1}};
      queue.submit([&](sycl::handler& cgh) {
        auto acc_result = buffer_result.get_access(cgh);
        sycl::local_accessor<int> t0{sycl::range<1>{1}, cgh};
        cgh.parallel_for<kernel_name_local<1>>(
            sycl::nd_range<1>{sycl::range<1>{1}, sycl::range<1>{1}},
            [=](sycl::nd_item<1> nd_item) {
              t0[0] = val;
              sycl::local_accessor<int> t1(t0);
              t0[0] = new_val;
              acc_result[0] = t1[0];
            });
      });
    }
    CHECK(new_val == result);
  }

  SECTION("mutation to original, const copy") {
    {
      sycl::buffer<int> buffer_result{&result, sycl::range<1>{1}};
      queue.submit([&](sycl::handler& cgh) {
        auto acc_result = buffer_result.get_access(cgh);
        sycl::local_accessor<int> t0{sycl::range<1>{1}, cgh};
        cgh.parallel_for<kernel_name_local<2>>(
            sycl::nd_range<1>{sycl::range<1>{1}, sycl::range<1>{1}},
            [=](sycl::nd_item<1> nd_item) {
              t0[0] = val;
              const sycl::local_accessor<int> t1(t0);
              t0[0] = new_val;
              acc_result[0] = t1[0];
            });
      });
    }
    CHECK(new_val == result);
  }
});

// Checks that every member function of sycl::local_accessor that the
// specification declares with a "noexcept" exception specification is actually
// declared that way.
namespace local_accessor_noexcept {

template <int Dimensions>
constexpr bool check_local_accessor() {
  using accessor_t = sycl::local_accessor<int, Dimensions>;

  CHECK_NOEXCEPT(std::declval<const accessor_t&>().byte_size());
  CHECK_NOEXCEPT(std::declval<const accessor_t&>().size());
  CHECK_NOEXCEPT(std::declval<const accessor_t&>().empty());
  CHECK_NOEXCEPT(std::declval<const accessor_t&>()
                     .template get_multi_ptr<sycl::access::decorated::yes>());
  CHECK_NOEXCEPT(std::declval<const accessor_t&>()
                     .template get_multi_ptr<sycl::access::decorated::no>());
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
  CHECK_NOEXCEPT(std::declval<const accessor_t&>().get_pointer());
#endif  // SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS

  return true;
}

static_assert(check_local_accessor<1>());
static_assert(check_local_accessor<2>());
static_assert(check_local_accessor<3>());

}  // namespace local_accessor_noexcept
