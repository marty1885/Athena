#Athena

`Athena` is a deep learning framework in C++. It is designed for maxima performance and the ablity to run on any systems without being vender locked-in

## Introduction
`Athena` provides
 * A computation backend that can be easily implemented
 * Tensor operators colosely resambles TensorFlow's
 * High level neural netowrk API

## Tensor and Backends
`Athena` handles tensor like how a game engine handles it's scenes and objecteds. The `Tensor` class does not creare nor operate a tensor itself. Instead it calls a `Backend` that does all the job.