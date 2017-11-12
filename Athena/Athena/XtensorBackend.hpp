#pragma once

#include <Athena/Backend.hpp>

#include <xtensor/xarray.hpp>

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

	virtual void* add(const void* handle1,const  void* handle2);
	virtual void* multiply(const void* handle1,const  void* handle2);
	virtual void* scalarMul(const  void* handle, float x);
	virtual void* scalarAdd(const void* handle, float val);
	virtual void selfScalarAdd(void* handle, float val);
	virtual void* div(const void* handle1,const  void* handle2);
	virtual void* subtract(const void* handle1,const  void* handle2);

	virtual void* dot(const void* handle1, const void* handle2);

	virtual void device(void* t, const float* ptr);
	virtual void host(void* t, float* ptr) const;

	virtual void* sum(const void* handle, const std::vector<size_t>& axis);
	virtual void* pow(const void* handle, float e);

	virtual void* sqrt(const void* handle);
	virtual void* abs(const void* handle);

	virtual std::vector<size_t> shape(void* handle) const;
	virtual void reshape(void* handle, const std::vector<size_t>& targetShape);
	virtual void* transpose(void* handle);
	virtual void* slice(void* handle, const std::vector<size_t>& begin, const std::vector<size_t>& size);
	virtual void* concatenate(const void* handle1, const void* handle2, int axis=0);
	virtual void* stack(const void* handle1, const void* handle2, int axis=0);

	virtual size_t size(const void* handle);

	static inline xt::xarray<float>& get(void* handle)
	{
		return *(xt::xarray<float>*)handle;
	}

	static inline const xt::xarray<float>& get(const void* handle)
	{
		return *(xt::xarray<float>*)handle;
	}


protected:

};

}
