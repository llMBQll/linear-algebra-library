# Linear Algebra Library

This library was created as part of my B.Sc. thesis at Warsaw University of Technology in 2021.<br>
It explores the possibilities provided by C++20 in regard to library design by designing a statically-typed linear
algebra library that supports multiple backends.<br>
This library allows the user to easily implement their own backends, even on per function basis. The implementation is
only required to be present only when it will be actually called, and this will be verified at compile time.<br>
Some features could not be tested as some features were not supported by all major compilers (GCC, Clang, MSVC) at the
time. This was the case for `std::source_location` and still is the case for module support.

## Features

- Multiple backends
    - CPU
    - CUDA
    - OpenCL
- Static typing, all implementations are resolved at compile time
- C++ standard library compatibility

## License

This project is licensed under the MIT, see [LICENSE](LICENSE) file for details.