#pragma once

#include <cstdint>

#define MBQ_SOURCE_LOCATION_CURRENT                                                                                    \
    ::mbq::SourceLocation                                                                                              \
    {                                                                                                                  \
        __LINE__, __FILE__, __func__                                                                                   \
    }

namespace mbq
{
    class SourceLocation
    {
    private:
        uint32_t _line;
        const char* _file;
        const char* _function;
    public:
        SourceLocation() = delete;
        constexpr SourceLocation(uint32_t line, const char* file, const char* function) noexcept
            : _line(line), _file(file), _function(function)
        { }
        constexpr SourceLocation(const SourceLocation&) noexcept = default;
        constexpr SourceLocation(SourceLocation&&) noexcept = default;
        constexpr SourceLocation& operator=(const SourceLocation&) noexcept = default;
        constexpr SourceLocation& operator=(SourceLocation&&) noexcept = default;

        [[nodiscard]] constexpr uint32_t line() const noexcept
        {
            return _line;
        }

        [[nodiscard]] constexpr const char* file_name() const noexcept
        {
            return _file;
        }

        [[nodiscard]] constexpr const char* function_name() const noexcept
        {
            return _function;
        }
    };
} // namespace mbq
