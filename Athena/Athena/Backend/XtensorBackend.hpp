#pragma once

#include <Athena/Backend/Backend.hpp>

#include <xtensor/xtensor_forward.hpp>

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

	virtual TensorImpl* createTensor(const std::vector<float>& vec, const Shape& shape) override;
	virtual TensorImpl* createTensor(const Shape& dims) override;
	TensorImpl* createTensor(const xt::xarray<float>& arr);
	virtual void destoryTensor(TensorImpl* handle) override;

	virtual TensorImpl* zeros(const Shape& shape) override;
	virtual TensorImpl* ones(const Shape& shape) override;
	virtual TensorImpl* rand(float lEdge, float rEdge, const Shape& shape) override;
	virtual TensorImpl* normal(float mean, float stddev, const Shape& shape) override;

protected:

};

}
