#include "random.cuh"

#include <bit>
#include <complex>
#include <random>

namespace mbq::cuda::detail
{
    struct MT19937
    {
        curandGenerator_t generator{nullptr};

        MT19937()
        {
            curandCreateGenerator(&generator, curandRngType_t::CURAND_RNG_PSEUDO_MT19937);
            curandSetPseudoRandomGeneratorSeed(generator, std::random_device{}());
        }

        ~MT19937()
        {
            curandDestroyGenerator(generator);
        }
    };

    MT19937& get_default_engine()
    {
        thread_local static MT19937 engine;
        return engine;
    }

    template <>
    curandStatus_t random(float* ptr, size_t count, float /*min*/, float /*max*/)
    {
        auto& engine = get_default_engine();
        return curandGenerateUniform(engine.generator, ptr, count);
    }

    template <>
    curandStatus_t random(double* ptr, size_t count, double /*min*/, double /*max*/)
    {
        auto& engine = get_default_engine();
        return curandGenerateUniformDouble(engine.generator, ptr, count);
    }

    template <>
    curandStatus_t random(std::complex<float>* ptr, size_t count, std::complex<float> /*min*/,
                          std::complex<float> /*max*/)
    {
        auto& engine = get_default_engine();
        // cast to float and multiply count times 2 to account for real and imaginary parts
        return curandGenerateUniform(engine.generator, std::bit_cast<float*>(ptr), count * 2);
    }

    template <>
    curandStatus_t random(std::complex<double>* ptr, size_t count, std::complex<double> /*min*/,
                          std::complex<double> /*max*/)
    {
        auto& engine = get_default_engine();
        // cast to double and multiply count times 2 to account for real and imaginary parts
        return curandGenerateUniformDouble(engine.generator, std::bit_cast<double*>(ptr), count * 2);
    }
} // namespace mbq::cuda::detail
