#pragma once

#include <vector>

namespace At
{

class Backend;

class TensorImpl
{
public:
	TensorImpl(Backend* backend) : backend_(backend) {}
	virtual ~TensorImpl(){}

	virtual void host(float* ptr) const = 0;
	virtual void device(const float* ptr) = 0;
	virtual size_t size() const = 0;
	virtual Shape shape() const = 0;
	virtual TensorImpl* clone() const = 0;

	virtual void add(float val) = 0;
	virtual void mul(float val) = 0;
	virtual void add(const TensorImpl* other) = 0;
	virtual void mul(const TensorImpl* other) = 0;
	virtual void subtract(const TensorImpl* other) = 0;
	virtual void divide(const TensorImpl* other) = 0;
	virtual void resize(const Shape& wantedShape) = 0;
	virtual void reciprocate() = 0;
	virtual TensorImpl* reshape(const Shape& wantedShape) const = 0;

	virtual TensorImpl* dot(const TensorImpl* other) const = 0;
	virtual TensorImpl* sqrt() const = 0;
	virtual TensorImpl* transpose() const = 0;
	virtual TensorImpl* transpose(const std::vector<intmax_t>& axis) const = 0;
	virtual TensorImpl* sum(intmax_t axis) const = 0;
	virtual TensorImpl* sum(const std::vector<intmax_t>& axis) const = 0;
	virtual TensorImpl* pow(float val) const = 0;
	virtual TensorImpl* slice(const Shape& begin, const Shape& size) const = 0;
	virtual TensorImpl* abs() const = 0;
	virtual TensorImpl* stack(const TensorImpl* other, int axis) const = 0;
	virtual TensorImpl* concatenate(const TensorImpl* other, int axis) const = 0;
	virtual TensorImpl* exp() const = 0;
	virtual TensorImpl* log() const = 0;

	virtual TensorImpl* graterThan(float val) const = 0;
	virtual TensorImpl* lesserThan(float val) const = 0;
	virtual TensorImpl* greaterOrEqual(float val) const = 0;
	virtual TensorImpl* lesserOrEqual(float val) const = 0;
	virtual TensorImpl* equalTo(float val) const = 0;


	virtual float* hostPtr() = 0;
	virtual const float* hostPtr() const = 0;


	inline Backend* backend()
	{
		return backend_;
	}

	inline const Backend* backend() const
	{
		return backend_;
	}

protected:
	Backend* backend_ = nullptr;

};

}
