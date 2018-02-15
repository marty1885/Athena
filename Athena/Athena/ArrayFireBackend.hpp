#pragma once

#include <arrayfire.h>

#include <Athena/Backend.hpp>

namespace At
{

class ArrayFireBackend : public Backend
{
public:
	ArrayFireBackend();
	virtual ~ArrayFireBackend()
	{
	}

	virtual TensorImpl* createTensor(const std::vector<float>& vec, const Shape& shape) override;
	virtual TensorImpl* createTensor(const Shape& dims) override;
	TensorImpl* createTensor(const af::array& arr, const Shape& s);
	virtual void destoryTensor(TensorImpl* handle) override;

	virtual TensorImpl* zeros(const Shape& shape) override;
	virtual TensorImpl* ones(const Shape& shape) override;
	virtual TensorImpl* rand(float lEdge, float rEdge, const Shape& shape) override;
	virtual TensorImpl* normal(float mean, float stddev, const Shape& shape) override;

protected:

};

}