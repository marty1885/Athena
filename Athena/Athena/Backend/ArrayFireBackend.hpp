#pragma once

#include <arrayfire.h>

#include <Athena/Backend/Backend.hpp>

namespace At
{


class ArrayFireBackend : public Backend
{
public:
	enum AFBackend
	{
		Default = 0,
		CPU = 1,
		CUDA = 2,
		OpenCL = 3
	};

	ArrayFireBackend(AFBackend afBackend=Default);
	virtual ~ArrayFireBackend()
	{
	}

	virtual TensorImpl* createTensor(const std::vector<float>& vec, const Shape& shape) override;
	virtual TensorImpl* createTensor(const std::vector<double>& vec, const Shape& shape);
	virtual TensorImpl* createTensor(const std::vector<int32_t>& vec, const Shape& shape);
	virtual TensorImpl* createTensor(const std::vector<int16_t>& vec, const Shape& shape);
	virtual TensorImpl* createTensor(const std::vector<bool>& vec, const Shape& shape);

	virtual TensorImpl* createTensor(const Shape& dims) override;
	TensorImpl* createTensor(const af::array& arr, const Shape& s);
	virtual void destoryTensor(TensorImpl* handle) override;

	virtual TensorImpl* zeros(const Shape& shape) override;
	virtual TensorImpl* ones(const Shape& shape) override;
	virtual TensorImpl* rand(float lEdge, float rEdge, const Shape& shape) override;
	virtual TensorImpl* normal(float mean, float stddev, const Shape& shape) override;

	void setAFBackend(AFBackend type);
	AFBackend getAFBackend() const;

protected:

};

}