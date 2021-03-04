#pragma once

#include "algorithm/fill.hpp"
#include "backends/opencl/exceptions/OpenCLException.hpp"
#include "backends/opencl/memory.hpp"

namespace mbq
{
    namespace detail
    {
        inline cl_int fill(cl_mem ptr, size_t offset, size_t size, const void* host_value, size_t value_size,
                           cl_command_queue queue)
        {
            return clEnqueueFillBuffer(queue, ptr, host_value, value_size, offset, size, 0, nullptr, nullptr);
        }
    } // namespace detail

    template <typename T>
    struct Fill<opencl::Allocator<T>>
    {
        using value_type = T;

        template <std::output_iterator<const value_type&> O, std::sentinel_for<O> S>
        constexpr O operator()(O first, S last, const value_type& value) const
        {
            auto state = first.get_allocator().state();
            auto [ptr, offset] = (&(*first)).get();
            auto res = detail::fill(ptr, offset * sizeof(T), (last - first) * sizeof(T), &value, sizeof(T),
                                    state->command_queue);
            if (res != CL_SUCCESS)
                MBQ_THROW_EXCEPTION(OpenCLException, res);
            return last;
        }
    };
} // namespace mbq