#pragma once

#include "backends/cuda/exceptions/CublasException.hpp"

#include <cublas_v2.h>
#include <exception>

namespace mbq::cuda
{
    class Context
    {
    public:
        cublasHandle_t handle{nullptr};
    public:
        Context()
        {
            auto status = cublasCreate_v2(&handle);
            if (status != cublasStatus_t::CUBLAS_STATUS_SUCCESS)
                MBQ_THROW_EXCEPTION(CublasException, status);
        }

        ~Context()
        {
            auto status = cublasDestroy_v2(handle);
            if (status != cublasStatus_t::CUBLAS_STATUS_SUCCESS)
                exit(1);
            handle = nullptr;
        }

        static Context* get_default()
        {
            static Context ctx;
            return &ctx;
        }
    };
} // namespace mbq::cuda