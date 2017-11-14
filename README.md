# Athena

`Athena` is a deep learning framework in C++. It is designed for maxima performance and the ablity to run on any systems and accelerators without being vender locked-in.

## Introduction

`Athena` provides

- A computation backend that can be easily implemented
- Tensor operators colsely resambles TensorFlow's
- High level neural netowrk API

## Tensor and Backends

`Athena` handles tensor like how a game engine handles shaders and meshes. The `Tensor` class does not creare nor operate a tensor itself. Instead it calls a `Backend` that does all the job. This design allows `Athena` to load backends at runtime and even utilize multiple backened at once.<br>
Backends in `Athena` can do more than performing tensor operations. So called `Algoritms` can be inserted into a backend then quary at runtime to perforem specific tasks if the uderlying library has a way to do it faster then using tensor operations.

Currently only the `XtensorBackend` using `xtensor` as the base library is avliable. Other backends using faster libraries like `ArrayFire`, `MIOpen` and `blaze` is planed.

## Examples

`example1` in the `examples` folder demostrates how to implement a MLP to solve the XOR problem in Athena.<br>
`mnist` contains an example using a MLP to classify MNIST digits.

## Development and contribution

Since deep learning and it's community is rapidly growing, we'd love to get contribution from you to accelerate Athena's development. Feel free to open a pull request or an issue to report a bug or request features.
