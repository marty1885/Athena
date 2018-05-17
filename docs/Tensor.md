Tensors in Athena are modeled after numpy's `ndarray`.

## Initializing a Tensor
A Tensor can be initalized from (nested) initalizer_list. Like so:
```C++
At::Tensor t = {{1,2,3},{1,2,3}};
```

Be aware that all initalizer_list in the same axis must have the same amount of elements. Otherwire an exceptions will be thrown.<br>
For example. This won't work.

```C++
//Throws an error
At::Tensor t = {{1,2,3},{1,2}};//Missing one value in the second vector
```

Also be aware that trying to initalize with the last axis having only 1 element will cause ambiguous constructor call. In that case, initalize the tensor with a valid method then reshape it.<br>
Ex:

```C++
//This won't compile
At::Tensor t = {{0},{1},{1},{0}};

//Do this
At::Tensor t = {{0,1,1,0}};
t = t.reshape({4,1});
```

## Getting the shape/volume of a Tensor
Getting the shape of a tensor can be done by calling the shape() method. Which when called. Will return a `Shape` variable (`vector<intmax_t>` with extra fratures). The volume of a tensor can be calculated by calling volume().
```C++
Shape s = tensor.shape();
intmax_t v = s.volume();//Or tensor.volume(), same thing
```

## Printing a Tensor
A tensor can be printed by std::ostream or converted into a std::string. Use the best way for your perpuse.

```C++
//Printing a tensor
std::cout << tensor << std::endl;

//Convert a tensor into a string
std::string = At::to_string(tensor);
```

## Reshaping a Tensor
Reshaping works the same way aas it works in numpy. The data it self isn't cahnged but the shape is.<br>
Tensor reshaping in Athena also supports solving a missing dimention. <br>
For example:<br>

```C++
At::Tensor t = At::ones({50,25});
At::Tensor q = t.reshape({At::Shape::None, 50});//Reshaped to {25,50}
```

Just be aware that at most 1 unknown dimention can be solve and the target shape must have the same volume as the original tensor. Else an exception is thrown.

## (nested) std::vector to Tensor
Converting (nested) vector into Tensor is easy.

```C++
std::vector<std::vector<float>> data = {{0,1,2}, {3,4,5}};
At::Tensor t = data;
```

Be aware like the case of initalizer_list, all vectors in the same axis must have the same amount of elements. Otherwire an exceptions will be thrown.<br>
For example. This won't work.

```C++
std::vector<std::vector<float>> data = {{0,1,2}, {2,3}};//Missing one value in the second vector
At::Tensor t = data; // throws an AtError
```

## Tensor to std::vector
There are no support for now to convert Tensors into ND vecrtors. But converting a Tensor into an 1D (flatten) vector is easy.<br>
This could be done by calling the `host()` (copy all data of the tensor from whatever it is to the host memory) method.

```C++
std::vector<float> vec = tensor.host();
```

## Xtensor Integration
With the `XtensorIntegration.hpp` header, `xarray` and `Tensor` can be converted with ease. This allows easy access to `npy` and `csv` data via xtensor. For example.
```C++
xt::xarray<float> data = xt::load_npy<float>("data.npy");
At::Tensor t = At::Tensor::from(data);

//Vise versa
xt::xarray<float> result = At::Tensor::to<xt::xarray<float>>(t);
```
Note that this only works with xarrays if types suppored by Tensor. Eg: `float`, `double`, `int`, `short` and `bool`.
