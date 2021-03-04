#pragma once

#define REAL(m_x) m_x.x
#define IMAG(m_x) m_x.y

#define MAKE_COMPLEX(m_type, m_complex_type, m_suffix)                                                                 \
    inline m_complex_type make_complex_##m_suffix(m_type real, m_type imag)                                            \
    {                                                                                                                  \
        m_complex_type z;                                                                                              \
        z.x = real;                                                                                                    \
        z.y = imag;                                                                                                    \
        return z;                                                                                                      \
    }

#define ADD(m_type, m_complex_type, m_suffix)                                                                          \
    inline m_complex_type add_##m_suffix(m_complex_type x, m_complex_type y)                                           \
    {                                                                                                                  \
        m_complex_type z;                                                                                              \
        z.x = REAL(x) + REAL(y);                                                                                       \
        z.y = IMAG(x) + IMAG(y);                                                                                       \
        return z;                                                                                                      \
    }

#define SUB(m_type, m_complex_type, m_suffix)                                                                          \
    inline m_complex_type sub_##m_suffix(m_complex_type x, m_complex_type y)                                           \
    {                                                                                                                  \
        m_complex_type z;                                                                                              \
        z.x = REAL(x) - REAL(y);                                                                                       \
        z.y = IMAG(x) - IMAG(y);                                                                                       \
        return z;                                                                                                      \
    }

#define MUL(m_type, m_complex_type, m_suffix)                                                                          \
    inline m_complex_type mul_##m_suffix(m_complex_type x, m_complex_type y)                                           \
    {                                                                                                                  \
        m_complex_type z;                                                                                              \
        z.x = REAL(x) * REAL(y) - IMAG(x) * IMAG(y);                                                                   \
        z.y = REAL(x) * IMAG(y) + IMAG(x) * REAL(y);                                                                   \
        return z;                                                                                                      \
    }

#define DIV(m_type, m_complex_type, m_suffix)                                                                          \
    inline m_complex_type div_##m_suffix(m_complex_type x, m_complex_type y)                                           \
    {                                                                                                                  \
        m_complex_type z;                                                                                              \
        m_type s = fabs(REAL(y)) + fabs(IMAG(y));                                                                      \
        m_type oos = 1.0 / s;                                                                                          \
        m_type ars = REAL(x) * oos;                                                                                    \
        m_type ais = IMAG(x) * oos;                                                                                    \
        m_type brs = REAL(y) * oos;                                                                                    \
        m_type bis = IMAG(y) * oos;                                                                                    \
        s = (brs * brs) + (bis * bis);                                                                                 \
        oos = 1.0 / s;                                                                                                 \
        z.x = ((ars * brs) + (ais * bis)) * oos;                                                                       \
        z.y = ((ais * brs) - (ars * bis)) * oos;                                                                       \
        return z;                                                                                                      \
    }

#define CONJ(m_type, m_complex_type, m_suffix)                                                                         \
    inline m_complex_type conj_##m_suffix(m_complex_type x)                                                            \
    {                                                                                                                  \
        m_complex_type z;                                                                                              \
        z.x = REAL(x);                                                                                                 \
        z.y = -IMAG(x);                                                                                                \
        return z;                                                                                                      \
    }

#define NORM(m_type, m_complex_type, m_suffix)                                                                         \
    inline m_type norm_##m_suffix(m_complex_type x)                                                                    \
    {                                                                                                                  \
        return REAL(x) * REAL(x) + IMAG(x) * IMAG(x);                                                                  \
    }

#define EXP(m_type, m_complex_type, m_suffix)                                                                          \
    inline m_complex_type exp_##m_suffix(m_complex_type x)                                                             \
    {                                                                                                                  \
        m_type e;                                                                                                      \
        m_complex_type z;                                                                                              \
        e = exp(REAL(x));                                                                                              \
        z.x = e * cos(IMAG(x));                                                                                        \
        z.y = e * sin(IMAG(x));                                                                                        \
        return z;                                                                                                      \
    }

#define LOG(m_type, m_complex_type, m_suffix)                                                                          \
    inline m_complex_type log_##m_suffix(m_complex_type x)                                                             \
    {                                                                                                                  \
        m_type log_abs = log(hypot(REAL(x), IMAG(x)));                                                                 \
        m_type theta = atan2(IMAG(x), REAL(x));                                                                        \
        return make_complex_##m_suffix(log_abs, theta);                                                                \
    }

#define POW_REAL(m_type, m_complex_type, m_suffix)                                                                     \
    inline m_complex_type pow_real_##m_suffix(m_complex_type x, m_type exponent)                                       \
    {                                                                                                                  \
        if (IMAG(x) == 0)                                                                                              \
        {                                                                                                              \
            if (signbit(IMAG(x)))                                                                                      \
                return conj_##m_suffix(make_complex_##m_suffix(pow(REAL(x), exponent), 0));                            \
            else                                                                                                       \
                return make_complex_##m_suffix(pow(REAL(x), exponent), 0);                                             \
        }                                                                                                              \
        else                                                                                                           \
            return exp_##m_suffix(mul_##m_suffix(make_complex_##m_suffix(exponent, 0), log_##m_suffix(x)));            \
    }

#define POW(m_type, m_complex_type, m_suffix)                                                                          \
    inline m_complex_type pow_##m_suffix(m_complex_type x, m_complex_type exponent)                                    \
    {                                                                                                                  \
        if (IMAG(x) == 0)                                                                                              \
            return pow_real_##m_suffix(x, REAL(exponent));                                                             \
        else if (IMAG(x) == 0 && REAL(x) > 0)                                                                          \
            return exp_##m_suffix(mul_##m_suffix(exponent, log(REAL(x))));                                             \
        else                                                                                                           \
            return exp_##m_suffix(mul_##m_suffix(exponent, log_##m_suffix(x)));                                        \
    }

#define SIN(m_type, m_complex_type, m_suffix)                                                                          \
    inline m_complex_type sin_##m_suffix(m_complex_type x)                                                             \
    {                                                                                                                  \
        m_complex_type z;                                                                                              \
        z.x = sin(REAL(x)) * cosh(IMAG(x));                                                                            \
        z.y = cos(REAL(x)) * sinh(IMAG(x));                                                                            \
        return z;                                                                                                      \
    }

#define SINH(m_type, m_complex_type, m_suffix)                                                                         \
    inline m_complex_type sinh_##m_suffix(m_complex_type x)                                                            \
    {                                                                                                                  \
        m_complex_type z;                                                                                              \
        z.x = sinh(REAL(x)) * cos(IMAG(x));                                                                            \
        z.y = cosh(REAL(x)) * sin(IMAG(x));                                                                            \
        return z;                                                                                                      \
    }

#define COS(m_type, m_complex_type, m_suffix)                                                                          \
    inline m_complex_type cos_##m_suffix(m_complex_type x)                                                             \
    {                                                                                                                  \
        m_complex_type z;                                                                                              \
        z.x = cos(REAL(x)) * cosh(IMAG(x));                                                                            \
        z.y = -sin(REAL(x)) * sinh(IMAG(x));                                                                           \
        return z;                                                                                                      \
    }

#define COSH(m_type, m_complex_type, m_suffix)                                                                         \
    inline m_complex_type cosh_##m_suffix(m_complex_type x)                                                            \
    {                                                                                                                  \
        m_complex_type z;                                                                                              \
        z.x = cosh(REAL(x)) * cos(IMAG(x));                                                                            \
        z.y = sinh(REAL(x)) * sin(IMAG(x));                                                                            \
        return z;                                                                                                      \
    }

#define TAN(m_type, m_complex_type, m_suffix)                                                                          \
    inline m_complex_type tan_##m_suffix(m_complex_type x)                                                             \
    {                                                                                                                  \
        m_complex_type s = sin_##m_suffix(x);                                                                          \
        m_complex_type c = cos_##m_suffix(x);                                                                          \
        return div_##m_suffix(s, c);                                                                                   \
    }

#define TANH(m_type, m_complex_type, m_suffix)                                                                         \
    inline m_complex_type tanh_##m_suffix(m_complex_type x)                                                            \
    {                                                                                                                  \
        m_complex_type s = sinh_##m_suffix(x);                                                                         \
        m_complex_type c = cosh_##m_suffix(x);                                                                         \
        return div_##m_suffix(s, c);                                                                                   \
    }

#define COMPLEX_IMPL(m_type, m_complex_type, m_suffix)                                                                 \
    MAKE_COMPLEX(m_type, m_complex_type, m_suffix)                                                                     \
    ADD(m_type, m_complex_type, m_suffix)                                                                              \
    SUB(m_type, m_complex_type, m_suffix)                                                                              \
    MUL(m_type, m_complex_type, m_suffix)                                                                              \
    DIV(m_type, m_complex_type, m_suffix)                                                                              \
    CONJ(m_type, m_complex_type, m_suffix)                                                                             \
    NORM(m_type, m_complex_type, m_suffix)                                                                             \
    EXP(m_type, m_complex_type, m_suffix)                                                                              \
    LOG(m_type, m_complex_type, m_suffix)                                                                              \
    POW_REAL(m_type, m_complex_type, m_suffix)                                                                         \
    POW(m_type, m_complex_type, m_suffix)                                                                              \
    SIN(m_type, m_complex_type, m_suffix)                                                                              \
    SINH(m_type, m_complex_type, m_suffix)                                                                             \
    COS(m_type, m_complex_type, m_suffix)                                                                              \
    COSH(m_type, m_complex_type, m_suffix)                                                                             \
    TAN(m_type, m_complex_type, m_suffix)                                                                              \
    TANH(m_type, m_complex_type, m_suffix)

#define COMPLEX_FLOAT COMPLEX_IMPL(float, float2, c)
#define COMPLEX_DOUBLE COMPLEX_IMPL(double, double2, z)