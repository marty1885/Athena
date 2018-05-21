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
	virtual ~ArrayFireBackend() = default;

	virtual TensorImpl* createTensor(const std::vector<float>& vec, const Shape& shape) override;
	virtual TensorImpl* createTensor(const std::vector<double>& vec, const Shape& shape) override;
	virtual TensorImpl* createTensor(const std::vector<int32_t>& vec, const Shape& shape) override;
	virtual TensorImpl* createTensor(const std::vector<int16_t>& vec, const Shape& shape) override;
	virtual TensorImpl* createTensor(const std::vector<bool>& vec, const Shape& shape) override;
	virtual TensorImpl* clone(const TensorImpl* handle) override;

	virtual TensorImpl* createTensor(const Shape& dims) override;
	TensorImpl* createTensor(const af::array& arr, const Shape& s);
	virtual void destoryTensor(TensorImpl* handle) override;

	virtual TensorImpl* zeros(const Shape& shape, DType dtype=DType::float32) override;
	virtual TensorImpl* ones(const Shape& shape, DType dtype=DType::float32) override;
	virtual TensorImpl* rand(float lEdge, float rEdge, const Shape& shape) override;
	virtual TensorImpl* normal(float mean, float stddev, const Shape& shape) override;

	virtual void eval(TensorImpl* impl) override;

	virtual Shape shape(const TensorImpl* impl) const override;
	virtual intmax_t size(const TensorImpl* impl) const override;
	virtual DType dtype(const TensorImpl* impl) const override;

	virtual void selfReciprocate(TensorImpl* impl) override;
	virtual void selfAdd(TensorImpl* impl, float val) override;
	virtual void selfMul(TensorImpl* impl, float val) override;
	virtual void selfAdd(TensorImpl* impl, const TensorImpl* other) override;
	virtual void selfMul(TensorImpl* impl, const TensorImpl* other) override;
	virtual void selfSub(TensorImpl* impl, const TensorImpl* other) override;
	virtual void selfDiv(TensorImpl* impl, const TensorImpl* other) override;

	virtual TensorImpl* sqrt(const TensorImpl* impl) override;
	virtual TensorImpl* abs(const TensorImpl* impl) override;
	virtual TensorImpl* exp(const TensorImpl* impl) override;
	virtual TensorImpl* log(const TensorImpl* impl) override;
	virtual TensorImpl* pow(const TensorImpl* impl, float val) override;

	virtual TensorImpl* dot(const TensorImpl* impl, const TensorImpl* other) override;

	virtual void modDims(TensorImpl* impl, const Shape& wantedShape) override;
	virtual TensorImpl* reshape(const TensorImpl* impl, const Shape& wantedShape) override;
	virtual TensorImpl* transpose(const TensorImpl* impl) override;
	//virtual TensorImpl* stack(const TensorImpl* impl, const TensorImpl* other, int axis) override;
	virtual TensorImpl* concatenate(const std::vector<TensorImpl const*>& arrs, int axis) override;
	virtual TensorImpl* chunk(const TensorImpl* impl, const Shape& begin, const Shape& size) override;

	virtual TensorImpl* sum(const TensorImpl* impl, intmax_t axis) override;
	//virtual TensorImpl* sum(const TensorImpl* impl, const std::vector<intmax_t>& axis) override;

	virtual void host(const TensorImpl* impl, float* ptr) const override;
	virtual void host(const TensorImpl* impl, double* ptr) const override;
	virtual void host(const TensorImpl* impl, int32_t* ptr) const override;
	virtual void host(const TensorImpl* impl, int16_t* ptr) const override;
	virtual void host(const TensorImpl* impl, bool* ptr) const override;

	virtual void device(TensorImpl* impl, const float* ptr) override;
	virtual void device(TensorImpl* impl, const double* ptr) override;
	virtual void device(TensorImpl* impl, const int32_t* ptr) override;
	virtual void device(TensorImpl* impl, const int16_t* ptr) override;
	virtual void device(TensorImpl* impl, const bool* ptr) override;

	// virtual void* hostPtr(TensorImpl* impl) override;
	// virtual const void* hostPtr(const TensorImpl* impl) override;

	virtual TensorImpl* greaterThan(const TensorImpl* impl,float val) override;
	virtual TensorImpl* lesserThan(const TensorImpl* impl,float val) override;
	virtual TensorImpl* greaterOrEqual(const TensorImpl* impl,float val) override;
	virtual TensorImpl* lesserOrEqual(const TensorImpl* impl,float val) override;
	virtual TensorImpl* equalTo(const TensorImpl* impl,float val) override;

	void setAFBackend(AFBackend type);
	AFBackend getAFBackend() const;

protected:

};

}