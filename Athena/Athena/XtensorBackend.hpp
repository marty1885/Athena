#pragma once

#include <Athena/Backend.hpp>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>

namespace At
{

//Implements a backend in xtensor
class XtensorBackend : public Backend
{
public:
	XtensorBackend();
	virtual ~XtensorBackend()
	{
	}

	virtual size_t createTensor(const xt::xarray<float>& arr);
	virtual size_t createTensor(const std::vector<size_t>& dims);
	virtual size_t createTensor(const std::vector<float>& vec, const std::vector<size_t>& shape);
	virtual size_t copyTensor(size_t src);
	virtual void destoryTensor(size_t handle);
	virtual size_t zeros(const std::vector<size_t>& shape);
	virtual size_t ones(const std::vector<size_t>& shape);
	virtual size_t rand(float lEdge, float rEdge, const std::vector<size_t>& shape);

	virtual size_t add(size_t handle1, size_t handle2);
	virtual size_t multiply(size_t handle1, size_t handle2);
	virtual size_t scalarMul(float x, size_t handle);
	virtual size_t scalarAdd(size_t handle, float val);
	virtual void selfScalarAdd(size_t handle, float val);
	virtual size_t div(size_t handle1, size_t handle2);
	virtual size_t subtract(size_t handle1, size_t handle2);

	virtual size_t dot(size_t handle1, size_t handle2);

	virtual std::vector<size_t> shape(size_t handle) const;
	virtual void reshape(size_t handle, const std::vector<size_t>& targetShape);
	virtual size_t transpose(size_t handle);
	virtual size_t slice(size_t handle, const std::vector<size_t>& begin, const std::vector<size_t>& size);

	virtual size_t sum(size_t handle, const std::vector<size_t>& axis);
	virtual size_t pow(size_t handle, float e) ;

	virtual void device(size_t handle, const float* ptr);
	virtual void host(size_t handle, float* ptr) const;

	inline const xt::xarray<float>& get(size_t handle) const
	{
		return storage_[handle];
	}

	inline xt::xarray<float>& get(size_t handle)
	{
		return storage_[handle];
	}


protected:
	std::vector<xt::xarray<float>> storage_;

};

}