# AI

> A neural network that can be trained on given datasets to predict the output of a given input.

## Requirements

- A compiler that supports C++20
- CMake 3.20 or higher
- [`libtorch`](https://pytorch.org/cppdocs/installing.html)
- [`cURL`](https://curl.se/download.html)
- [`libzip`](https://libzip.org/download/)

## Quick Start

- Clone the repository

```bash
$ git clone https://github.com/Sid110307/AI.git
```

- Configure CMake

```bash
$ cd AI
$ cmake -S . -B bin
```

- Build the project

```bash
$ cmake --build bin --target all -j4
```

- Run the project

```bash
$ ./bin/AI
```

## License

[MIT](https://opensource.org/licenses/MIT)