# ICICLE for Gnark

> [!WARNING]
> This is a standalone version of ICICLE specifically meant for integration with the Gnark codebase. It is **NOT** meant to be used together with the main ICICLE repo.
> 
> If you are looking for the main ICICLE repo, you can find it [here](https://github.com/ingonyama-zk/icicle)

<div align="center">
  ICICLE is a high-performance cryptographic acceleration library designed to optimize cryptographic computations across various hardware platforms, including CPUs, GPUs, and other accelerators.
</div>

<p align="center">
  <br>
  <img alt="ICICLE" width="300" height="300" src="assets/McPaw.gif"/>
</p>

<p align="center">
  <br>
  <br>
  <a href="https://discord.gg/EVVXTdt6DF">
    <img src="https://img.shields.io/discord/1063033227788423299?logo=discord&style=flat&labelColor=black&color=5765f2" alt="Chat with us on Discord">
  </a>
  <a href="https://www.linkedin.com/company/ingonyama">
    <img src="https://img.shields.io/badge/follow-blue?style=flat&logo=linkedin&labelColor=black" alt="Follow us on LinkedIn">
  </a>
  <a href="https://x.com/Ingo_zk">
    <img src="https://img.shields.io/badge/follow-darkgray?style=flat&logo=x&labelColor=black" alt="Follow us on Twitter">
  </a>
</p>

## Background

Zero Knowledge Proofs (ZKPs) are considered one of the greatest achievements of modern cryptography. Accordingly, ZKPs are expected to disrupt a number of industries and will usher in an era of trustless and privacy preserving services and infrastructure.

We believe that ICICLE will be a cornerstone in the acceleration of ZKPs:

- **Versatility**: Supports multiple hardware platforms, making it adaptable to various computational environments.
- **Efficiency:** Designed to leverage the parallel nature of ZK computations, whether on GPUs, CPUs, or other accelerators.
- **Scalability:** Provides an easy-to-use and scalable solution for developers, allowing them to optimize cryptographic operations with minimal effort.

## Getting Started

### Prerequisites

- Any compatible hardware: CPU or Nvidia GPU.
- [CMake](https://cmake.org/files/), Version 3.18 or above. Latest version recommended
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads), Required only if using NVIDIA GPUs (version 12.0 or newer).
- Clang
- User permissions to create directories in GOMODCACHE
- User permissions to install libraries in `/usr/local/lib`

> [!NOTE]
> For older GPUs that only support CUDA 11, ICICLE may still function, but official support is for CUDA 12 onwards.

### Building from source

Gnark includes icicle-gnark as a direct dependency. Before gnark can use icicle-gnark, the c++ libraries need to be built. Follow these instructions to build the libs.

1. Run `go mod tidy` to pull all the dependecies.
2. Navigate to the icicle-gnark dependency in your GOMODCACHE and build the c++ libraries there.

    ```sh
    cd $(go env GOMODCACHE)/github.com/ingonyama-zk/icicle-gnark/v3@<version>/wrappers/golang
    chmod +x build.sh
    ./build.sh -curve=bn254 
    ```

3. Double check that the following frontend libs are located in `/usr/local/lib`.
   1. libicicle_curve_bn254.so
   2. libicicle_field_bn254.so
   3. libicicle_device.so
4. Double check that the following backend libs are located in `/usr/local/lib/backend/**`.
   1. libicicle_backend_cuda_curve_bn254.so
   2. libicicle_backend_cuda_field_bn254.so
   3. libicicle_backend_cuda_device.so
5. Finally, export the `ICICLE_BACKEND_INSTALL_DIR` environment variable to the location with the backend libraries

   ```sh
    export ICICLE_BACKEND_INSTALL_DIR=/usr/local/lib/backend
   ```

> [!TIP]
> By default, the above steps build both the CPU and CUDA backends. You can switch the device by using the CreateDevice API.

> [!NOTE]
> To specify a curve, use the flag -curve=\<curve>, where \<curve> can be one of the following: bn254, bls12_377, bls12_381, or bw6_761.
>
> **Currently, Gnark only has support for bn254 with icicle-gnark**

## Contributions

Join our [Discord Server][DISCORD] and find us on the ICICLE channel. We will be happy to work together to support your use case, and talk features, bugs and design.

### Development Contributions

Please see the [main ICICLE repo](https://github.com/ingonyama-zk/icicle) for contributing.

## Help & Support

For help and support talk to our devs in our discord channel [#ICICLE](https://discord.gg/EVVXTdt6DF) or contact us at <support@ingonyama.com>.

## License

This version of ICICLE is distributed under the terms of the MIT License.

See [LICENSE-MIT][LMIT] for details.

<!-- Begin Links -->
[LMIT]: ./LICENSE
[DISCORD]: https://discord.gg/Y4SkbDf2Ff
<!-- End Links -->
