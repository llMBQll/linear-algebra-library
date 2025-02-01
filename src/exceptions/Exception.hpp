#pragma once

#include <exception>
#include <source_location>

#define MBQ_MAKE_EXCEPTION(m__type, ...)                                                                               \
    m__type                                                                                                            \
    {                                                                                                                  \
        __VA_ARGS__                                                                                                    \
    }

#define MBQ_THROW_EXCEPTION(m__type, ...) ::mbq::throw_exception(MBQ_MAKE_EXCEPTION(m__type, __VA_ARGS__))

#ifndef MBQ_EXCEPTIONS
    #define MBQ_EXCEPTIONS 1
#endif

namespace mbq
{
#if MBQ_EXCEPTIONS
    template <typename E>
    [[noreturn]] inline void throw_exception(const E& e)
    {
        throw e;
    }
#else
    [[noreturn]] inline void throw_exception(const std::exception& e);
#endif

    class Exception : public std::exception
    {
    private:
        const char* _message;
        std::source_location _location;
    public:
        explicit Exception(const std::source_location& location = {}) : _message("mbq::Exception"), _location(location)
        { }

        explicit Exception(const char* message, const std::source_location& location = {})
            : _message(message), _location(location)
        { }

        Exception(const Exception&) noexcept = default;

        ~Exception() noexcept override = default;

        [[nodiscard]] const char* what() const noexcept override
        {
            return _message;
        }

        [[nodiscard]] const std::source_location& location() const noexcept
        {
            return _location;
        }
    };

    inline std::string to_string(const Exception& e)
    {
        return std::string{e.location().file_name()} + "(" + std::to_string(e.location().line()) + ")[" +
               e.location().function_name() + "] " + e.what();
    }

    inline std::ostream& operator<<(std::ostream& out, const Exception& e)
    {
        out << e.location().file_name() << '(' << e.location().line() << ") [" << e.location().function_name() << "] "
            << e.what();
        return out;
    }
} // namespace mbq