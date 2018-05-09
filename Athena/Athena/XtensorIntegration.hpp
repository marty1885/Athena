#pragma once

#include <xtensor/xarray.hpp>

#include <Athena/Tensor.hpp>

namespace At
{

//TODO: Make accept different types
template<>
inline Tensor Tensor::from(const xt::xarray<float>& arr)
{
	auto shape = arr.shape();
	Shape s(shape.begin(), shape.end());

	//TODO: Anailate this copying of data
	std::vector<float> v(arr.size());
	for(size_t i=0;i<arr.size();i++)
		v[i] = arr[i];
	return At::Tensor(v, s);
}

template <> 
inline xt::xarray<float> Tensor::to(const Tensor& t)
{
	auto data = t.host();
	auto shape = t.shape();
	xt::xarray<float>::shape_type s(shape.begin(), shape.end());
	xt::xarray<float> arr = xt::xarray<float>::from_shape(s);
	for(size_t i=0;i<data.size();i++)
		arr[i] = data[i];
	return arr;
}

}
