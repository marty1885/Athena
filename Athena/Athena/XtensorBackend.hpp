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

	virtual void* createTensor(const std::vector<float>& vec, const std::vector<size_t>& shape);
	virtual void* createTensor(const std::vector<size_t>& dims);
	void* createTensor(const xt::xarray<float>& arr);
	virtual void* copyTensor(const void* src);
	virtual void destoryTensor(void* handle);
	virtual void* zeros(const std::vector<size_t>& shape);
	virtual void* ones(const std::vector<size_t>& shape);
	virtual void* rand(float lEdge, float rEdge, const std::vector<size_t>& shape);

	virtual void* add(void* handle1, void* handle2);
	virtual void* multiply(void* handle1, void* handle2);
	virtual void* scalarMul(float x, void* handle);
	virtual void* scalarAdd(void* handle, float val);
	virtual void selfScalarAdd(void* handle, float val);
	virtual void* div(void* handle1, void* handle2);
	virtual void* subtract(void* handle1, void* handle2);

	virtual void* dot(void* handle1, void* handle2);

	virtual void device(void* t, const float* ptr);
	virtual void host(void* t, float* ptr) const;

	virtual void* sum(void* handle, const std::vector<size_t>& axis);
	virtual void* pow(void* handle, float e);

	virtual std::vector<size_t> shape(void* handle) const;
	virtual void reshape(void* handle, const std::vector<size_t>& targetShape);
	virtual void* transpose(void* handle);
	virtual void* slice(void* handle, const std::vector<size_t>& begin, const std::vector<size_t>& size);

	virtual size_t size(void* handle);

	inline xt::xarray<float>& get(void* handle) const
	{
		return *(xt::xarray<float>*)handle;
	}

	inline const xt::xarray<float>& get(const void* handle) const
	{
		return *(xt::xarray<float>*)handle;
	}


protected:

};

}