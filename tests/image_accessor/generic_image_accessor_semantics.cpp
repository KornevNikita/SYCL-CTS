/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/semantics_reference.h"
#include "../image/default_image.h"
#include <cstddef>
#include <utility>

struct storage_sampled {
  std::size_t size;

  template <typename DataT, int Dimensions, sycl::image_target AccessTarget>
  explicit storage_sampled(
      const sycl::sampled_image_accessor<DataT, Dimensions, AccessTarget>&
          sampled_image_accessor)
      : size(sampled_image_accessor.size()) {}

  template <typename DataT, int Dimensions, sycl::image_target AccessTarget>
  bool check(
      const sycl::sampled_image_accessor<DataT, Dimensions, AccessTarget>&
          sampled_image_accessor) const {
    return sampled_image_accessor.size() == size;
  }
};

TEST_CASE("sampled_image_accessor common reference semantics (host_task)",
          "[sampled_image_accessor]") {
  auto sampled_image_0 = default_sampled_image::get();
  auto sampled_image_1 = default_sampled_image::get();
  sycl_cts::util::get_cts_object::queue()
      .submit([&](sycl::handler& cgh) {
        auto sampled_image_accessor_0 =
            default_sampled_image::get_acc<sycl::image_target::host_task>(
                sampled_image_0, cgh);
        auto sampled_image_accessor_1 =
            default_sampled_image::get_acc<sycl::image_target::host_task>(
                sampled_image_1, cgh);
        cgh.host_task([=] {
          auto sampled_image_accessor_0_copy = sampled_image_accessor_0;
          common_reference_semantics::check_host<storage_sampled>(
              sampled_image_accessor_0_copy, sampled_image_accessor_1,
              "sampled_image_accessor<int4, 1, image_target::host_task>");
        });
      })
      .wait_and_throw();
}

TEST_CASE("sampled_image_accessor common reference semantics (kernel)",
          "[sampled_image_accessor]") {
  auto sampled_image = default_sampled_image::get();
  using type = default_sampled_image::acc_type<sycl::image_target::device>;
  using data_type = default_sampled_image::data_type;
  common_reference_semantics::check_kernel<storage_sampled, type>(
      [&](sycl::handler& cgh) {
        return sampled_image.get_access<data_type>(cgh);
      },
      "sampled_image_accessor<int4, 1, image_target::device>",
      {sycl::aspect::image});
}

struct storage_unsampled {
  std::size_t size;

  template <typename DataT, int Dimensions, sycl::access_mode AccessMode,
            sycl::image_target AccessTarget>
  explicit storage_unsampled(const sycl::unsampled_image_accessor<
                             DataT, Dimensions, AccessMode, AccessTarget>&
                                 unsampled_image_accessor)
      : size(unsampled_image_accessor.size()) {}

  template <typename DataT, int Dimensions, sycl::access_mode AccessMode,
            sycl::image_target AccessTarget>
  bool check(const sycl::unsampled_image_accessor<DataT, Dimensions, AccessMode,
                                                  AccessTarget>&
                 unsampled_image_accessor) const {
    return unsampled_image_accessor.size() == size;
  }
};

TEST_CASE("unsampled_image_accessor common reference semantics (host_task)",
          "[unsampled_image_accessor]") {
  auto unsampled_image_0 = default_unsampled_image::get();
  auto unsampled_image_1 = default_unsampled_image::get();
  sycl_cts::util::get_cts_object::queue()
      .submit([&](sycl::handler& cgh) {
        auto unsampled_image_accessor_0 =
            default_unsampled_image::get_acc<sycl::access_mode::read,
                                             sycl::image_target::host_task>(
                unsampled_image_0, cgh);

        auto unsampled_image_accessor_1 =
            default_unsampled_image::get_acc<sycl::access_mode::read,
                                             sycl::image_target::host_task>(
                unsampled_image_1, cgh);
        cgh.host_task([=] {
          auto unsampled_image_accessor_0_copy = unsampled_image_accessor_0;
          common_reference_semantics::check_host<storage_unsampled>(
              unsampled_image_accessor_0_copy, unsampled_image_accessor_1,
              "unsampled_image_accessor<int4, 1, access_mode::read, "
              "image_target::host_task>");
        });
      })
      .wait_and_throw();
}

TEST_CASE("unsampled_image_accessor common reference semantics (kernel)",
          "[unsampled_image_accessor]") {
  auto unsampled_image = default_unsampled_image::get();
  using type = default_unsampled_image::acc_type<sycl::access_mode::read,
                                                 sycl::image_target::device>;
  using data_type = default_unsampled_image::data_type;
  common_reference_semantics::check_kernel<storage_unsampled, type>(
      [&](sycl::handler& cgh) {
        return unsampled_image.get_access<data_type, sycl::access_mode::read>(
            cgh);
      },
      "unsampled_image_accessor<int4, 1, access_mode::read, "
      "image_target::device>",
      {sycl::aspect::image});
}

// Checks that every member function of the image accessor classes that the
// specification declares with a "noexcept" exception specification is actually
// declared that way.
namespace image_accessor_noexcept {

// Coordinate type used by unsampled image accessors, see the specification of
// unsampled_image_accessor::read().
template <int Dimensions>
using unsampled_coord_t = std::conditional_t<
    Dimensions == 1, int,
    std::conditional_t<Dimensions == 2, sycl::int2, sycl::int4>>;

// Coordinate type used by sampled image accessors, see the specification of
// sampled_image_accessor::read().
template <int Dimensions>
using sampled_coord_t = std::conditional_t<
    Dimensions == 1, float,
    std::conditional_t<Dimensions == 2, sycl::float2, sycl::float4>>;

template <int Dimensions, sycl::image_target AccessTarget>
constexpr bool check_unsampled_image_accessor() {
  using data_t = sycl::int4;
  using accessor_t =
      sycl::unsampled_image_accessor<data_t, Dimensions,
                                     sycl::access_mode::read, AccessTarget>;
  using coord_t = unsampled_coord_t<Dimensions>;

  CHECK_NOEXCEPT(std::declval<const accessor_t&>().size());
  CHECK_NOEXCEPT(
      std::declval<const accessor_t&>().read(std::declval<const coord_t&>()));

  return true;
}

template <int Dimensions>
constexpr bool check_host_unsampled_image_accessor() {
  using data_t = sycl::int4;
  using accessor_t =
      sycl::host_unsampled_image_accessor<data_t, Dimensions,
                                          sycl::access_mode::read>;
  using coord_t = unsampled_coord_t<Dimensions>;

  CHECK_NOEXCEPT(std::declval<const accessor_t&>().size());
  CHECK_NOEXCEPT(
      std::declval<const accessor_t&>().read(std::declval<const coord_t&>()));

  return true;
}

template <int Dimensions, sycl::image_target AccessTarget>
constexpr bool check_sampled_image_accessor() {
  using data_t = sycl::int4;
  using accessor_t =
      sycl::sampled_image_accessor<data_t, Dimensions, AccessTarget>;
  using coord_t = sampled_coord_t<Dimensions>;

  CHECK_NOEXCEPT(std::declval<const accessor_t&>().size());
  CHECK_NOEXCEPT(
      std::declval<const accessor_t&>().read(std::declval<const coord_t&>()));

  return true;
}

template <int Dimensions>
constexpr bool check_host_sampled_image_accessor() {
  using data_t = sycl::int4;
  using accessor_t = sycl::host_sampled_image_accessor<data_t, Dimensions>;
  using coord_t = sampled_coord_t<Dimensions>;

  CHECK_NOEXCEPT(std::declval<const accessor_t&>().size());
#if !SYCL_CTS_COMPILING_WITH_DPCPP
  // FIXME: re-enable once DPC++ declares this overload of read() noexcept.
  CHECK_NOEXCEPT(
      std::declval<const accessor_t&>().read(std::declval<const coord_t&>()));
#endif

  return true;
}

template <int Dimensions>
constexpr bool check_all() {
  return check_unsampled_image_accessor<Dimensions,
                                        sycl::image_target::device>() &&
         check_unsampled_image_accessor<Dimensions,
                                        sycl::image_target::host_task>() &&
         check_host_unsampled_image_accessor<Dimensions>() &&
         check_sampled_image_accessor<Dimensions,
                                      sycl::image_target::device>() &&
         check_sampled_image_accessor<Dimensions,
                                      sycl::image_target::host_task>() &&
         check_host_sampled_image_accessor<Dimensions>();
}

static_assert(check_all<1>());
static_assert(check_all<2>());
static_assert(check_all<3>());

}  // namespace image_accessor_noexcept
