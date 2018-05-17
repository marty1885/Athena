#pragma once

#include <xtensor/xarray.hpp>

#include <Athena/Tensor.hpp>

namespace At
{

template <typename T>
inline Tensor __xtToTensor(const xt::xarray<T>& arr)
{
	auto shape = arr.shape();
	Shape s(shape.begin(), shape.end());

	//TODO: Annihilate this copying of data
	std::vector<T> v(arr.size());
	for(size_t i=0;i<arr.size();i++)
		v[i] = arr[i];
	return Tensor(v, s);
}

template <typename T>
inline xt::xarray<T> __TensorToXt(const Tensor& t)
{
	auto data = t.host<T>();
	auto shape = t.shape();
	typename xt::xarray<T>::shape_type s(shape.begin(), shape.end());
	xt::xarray<T> arr = xt::xarray<T>::from_shape(s);
	for(size_t i=0;i<data.size();i++)
		arr[i] = data[i];
	return arr;
}

template<>
inline Tensor Tensor::from(const xt::xarray<float>& arr)
{
	return __xtToTensor(arr);
}

template<>
inline Tensor Tensor::from(const xt::xarray<double>& arr)
{
	return __xtToTensor(arr);
}

template<>
inline Tensor Tensor::from(const xt::xarray<int32_t>& arr)
{
	return __xtToTensor(arr);
}

template<>
inline Tensor Tensor::from(const xt::xarray<int16_t>& arr)
{
	return __xtToTensor(arr);
}

template<>
inline Tensor Tensor::from(const xt::xarray<bool>& arr)
{
	return __xtToTensor(arr);
}

template <>
inline xt::xarray<float> Tensor::to(const Tensor& t)
{
	return __TensorToXt<float>(t);
}

template <>
inline xt::xarray<double> Tensor::to(const Tensor& t)
{
	return __TensorToXt<double>(t);
}

template <>
inline xt::xarray<int32_t> Tensor::to(const Tensor& t)
{
	return __TensorToXt<int32_t>(t);
}

template <>
inline xt::xarray<int16_t> Tensor::to(const Tensor& t)
{
	return __TensorToXt<int16_t>(t);
}

template <>
inline xt::xarray<bool> Tensor::to(const Tensor& t)
{
	return __TensorToXt<bool>(t);
}


}
