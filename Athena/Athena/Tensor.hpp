#pragma once

#include <Athena/Backend.hpp>

#include <assert.h>

#include <vector>
#include <numeric>
#include <iostream>

namespace At
{

class ReferenceCounter
{
protected:
	size_t count_;
public:
	ReferenceCounter(size_t initVal = 0)
		: count_(initVal)
	{
	}

	void addRef()
	{
		count_++;
	}
    
	int release()
	{
	    return --count_;
	}
};

class Tensor
{
public:
	Tensor()
	{
		referenceCounter_ = new ReferenceCounter(0);
		referenceCounter_->addRef();
	}

	Tensor(const std::vector<size_t>& shape, Backend* backend)
		: Tensor()
	{
		backend_ = backend;
		handle_ = backend->createTensor(shape);
	}
	
	Tensor(size_t handle, Backend* backend)
		: Tensor()
	{
		handle_ = handle;
		backend_ = backend;
	}

	Tensor(const std::vector<float>& vec, const std::vector<size_t>& shape, Backend* backend)
		: Tensor()
	{	
		backend_ = backend;
		handle_ = backend->createTensor(vec, shape);
	}

	Tensor(const Tensor& t)
		:referenceCounter_(t.referenceCounter())
	{
		//Workarround someone trying to copy a not initialized Tensor
		if(t.backend() == nullptr)
			return;

		backend_ = t.backend();
		handle_ = t.internalHandle();
		if(referenceCounter_ != nullptr)
			referenceCounter_->addRef();
	}
	
	Tensor& operator= (const Tensor& other)
	{
		if(other.backend() == nullptr)
			return *this;

		backend_ = other.backend();
		handle_ = other.internalHandle();
		referenceCounter_ = other.referenceCounter();
		if(referenceCounter_ != nullptr)
			referenceCounter_->addRef();

		return *this;
	}

	Tensor(Tensor&& t)
	{
		if(t.referenceCounter() == nullptr)
			return;

		referenceCounter_ = t.referenceCounter();
		backend_ = t.backend();
		handle_ = t.internalHandle();
		t.internalHandle() = 0;
		t.setReferenceCounter(nullptr);
	}

	Backend* backend() const
	{
		return backend_;
	}

	Tensor operator+(const Tensor& other) const
	{
		assert(other.backend() == backend());
		return Tensor(backend_->add(handle_, other.internalHandle()), backend_);
	}

	Tensor operator-(const Tensor& other) const
	{
		assert(other.backend() == backend());
		return Tensor(backend_->subtract(handle_, other.internalHandle()), backend_);
	}

	Tensor operator*(const Tensor& other) const
	{
		assert(other.backend() == backend());
		return Tensor(backend_->multiply(handle_, other.internalHandle()), backend_);
	}

	Tensor operator/(const Tensor& other) const
	{
		assert(other.backend() == backend());
		return Tensor(backend_->subtract(handle_, other.internalHandle()), backend_);
	}
	
	Tensor operator*(float amp) const
	{
		return Tensor(backend_->scalarMul(amp, handle_), backend_);
	}

	Tensor operator+(const float& x) const
	{
		assert(other.backend() == backend());
		return Tensor(backend_->scalarAdd(handle_, x), backend_);
	}

	Tensor operator-(const float& x) const
	{
		assert(other.backend() == backend());
		return Tensor(backend_->scalarAdd(handle_, -x), backend_);
	}

	Tensor slice(const std::vector<size_t>& begin, const std::vector<size_t>& size) const
	{
		return Tensor(backend_->slice(handle_, begin, size), backend_);
	}

	Tensor transpose() const
	{
		return Tensor(backend_->transpose(handle_), backend_);
	}

	Tensor sum(const std::vector<size_t>& axis)
	{
		return Tensor(backend_->sum(handle_, axis), backend_);
	}

	const std::vector<size_t> shape() const
	{
		return backend_->shape(handle_);
	}
	
	void operator-=(const Tensor& other)
	{
		//Optimize this
		*this = (*this - other);
	}

	void operator-=(const float& x)
	{
		backend_->selfScalarAdd(handle_,-x);
	}

	void reshape(const std::vector<size_t>& s)
	{
		backend_->reshape(handle_, s);
	}

	size_t size() const
	{
		auto& s = shape();
		return std::accumulate(s.begin(), s.end(), 1, std::multiplies<size_t>());
	}

	const size_t& internalHandle() const
	{
		return handle_;
	}

	size_t& internalHandle()
	{
		return handle_;
	}

	void host(float* ptr) const
	{
		backend_->host(handle_, ptr);
	}

	virtual ~Tensor()
	{

		if(referenceCounter_ != nullptr)
		{
			if(referenceCounter_->release() == 0)
			{
				if(handle_ != 0)
					backend_->destoryTensor(handle_);
				delete referenceCounter_;
			}
		}
	
		handle_ = 0;
		backend_ = nullptr;
	}

protected:
	ReferenceCounter* referenceCounter() const
	{
		return referenceCounter_;
	}

	void setReferenceCounter(ReferenceCounter* ptr)
	{
		referenceCounter_ = ptr;
	}

	Backend* backend_ = nullptr;
	size_t handle_ = 0;
	ReferenceCounter* referenceCounter_ = nullptr;
};

Tensor rand(float lEdge, float rEdge, const std::vector<size_t>& shape, Backend* backend)
{
	return Tensor(backend->rand(rEdge, lEdge, shape), backend);
}

Tensor zeros(const std::vector<size_t>& shape, Backend* backend)
{
	return Tensor(backend->zeros(shape), backend);
}

Tensor dot(const Tensor& a, const Tensor& b)
{
	return Tensor(a.backend()->dot(a.internalHandle(), b.internalHandle()), a.backend());
}

std::ostream& operator<< (std::ostream& os, const Tensor& t)
{
	std::vector<float> v(t.size());
	t.host(&v[0]);
	os << "{";
	for(auto val : v)
		os << val << ", ";
	os << "}";
	return os;
}

}
