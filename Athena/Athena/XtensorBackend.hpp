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

	virtual TensorImpl* createTensor(const std::vector<float>& vec, const Shape& shape);
	virtual TensorImpl* createTensor(const Shape& dims);
	virtual TensorImpl* createTensor(const xt::xarray<float>& arr);
	virtual void destoryTensor(TensorImpl* handle);

	virtual TensorImpl* zeros(const Shape& shape);
	virtual TensorImpl* ones(const Shape& shape);
	virtual TensorImpl* rand(float lEdge, float rEdge, const Shape& shape);
	virtual TensorImpl* normal(float mean, float stddev, const Shape& shape);

protected:

};

}
