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

	virtual void device(void* t, const float* ptr);
	virtual void host(void* t, float* ptr) const;

	virtual size_t size(void* handle);
	virtual void reshape(void* handle, const std::vector<size_t>& s);

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