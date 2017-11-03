#pragma once

#include <Athena/Backend.hpp>
#include <Athena/ReferenceCounter.hpp>

#include <assert.h>

#include <vector>
#include <numeric>
#include <iostream>

namespace At
{

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
	
	Tensor(void* handle, Backend* backend)
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
		handle_ = const_cast<void*>(t.internalHandle());
		if(referenceCounter_ != nullptr)
			referenceCounter_->addRef();
	}
	
	Tensor& operator= (const Tensor& other)
	{
		if(other.backend() == nullptr)
			return *this;

		backend_ = other.backend();
		handle_ = const_cast<void*>(other.internalHandle());
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

	inline Backend* backend() const
	{
		return backend_;
	}

	Tensor slice(const std::vector<size_t>& begin, const std::vector<size_t>& size) const
	{
		return Tensor(backend_->slice(handle_, begin, size), backend_);
	}

	Tensor transpose() const
	{
		return Tensor(backend_->transpose(handle_), backend_);
	}

	Tensor clone() const
	{
		return Tensor(backend_->copyTensor(handle_), backend_);
	}

	Tensor sum(const std::vector<size_t>& axis)
	{
		return Tensor(backend_->sum(handle_, axis), backend_);
	}

	Tensor pow(float e)
	{
		return Tensor(backend_->pow(handle_, e), backend_);
	}

	const std::vector<size_t> shape() const
	{
		return backend_->shape(handle_);
	}

	void reshape(const std::vector<size_t>& s)
	{
		backend_->reshape(handle_, s);
	}

	size_t size() const
	{
		return backend_->size(handle_);
	}

	inline const void* internalHandle() const
	{
		return handle_;
	}

	inline void*& internalHandle()
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
	inline ReferenceCounter* referenceCounter() const
	{
		return referenceCounter_;
	}

	inline void setReferenceCounter(ReferenceCounter* ptr)
	{
		referenceCounter_ = ptr;
	}

	Backend* backend_ = nullptr;
	ReferenceCounter* referenceCounter_ = nullptr;
	void* handle_ = nullptr;
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

Tensor operator+(const Tensor& t, const Tensor& other)
{
	assert(t.backend() == t.backend());
	return Tensor(t.backend()->add(t.internalHandle(), other.internalHandle()), t.backend());
}

Tensor operator-(const Tensor& t, const Tensor& other)
{
	assert(t.backend() == t.backend());
	return Tensor(t.backend()->subtract(t.internalHandle(), other.internalHandle()), t.backend());
}

void operator-=(Tensor& t, const Tensor& other)
{
	t = std::move(t-other);
}

Tensor operator*(const Tensor& t, const Tensor& other)
{
	assert(t.backend() == t.backend());
	return Tensor(t.backend()->multiply(t.internalHandle(), other.internalHandle()), t.backend());
}

Tensor operator/(const Tensor& t, const Tensor& other)
{
	assert(t.backend() == t.backend());
	return Tensor(t.backend()->div(t.internalHandle(), other.internalHandle()), t.backend());
}

void operator-=(Tensor& t, const float& x)
{
	t.backend()->selfScalarAdd(t.internalHandle(),-x);
}

Tensor operator+(const Tensor& t, float val)
{
	assert(t.backend() == t.backend());
	return Tensor(t.backend()->scalarAdd(t.internalHandle(), val), t.backend());
}

Tensor operator-(const Tensor& t, float val)
{
	assert(t.backend() == t.backend());
	return Tensor(t.backend()->scalarAdd(t.internalHandle(), -val), t.backend());
}

Tensor operator*(const Tensor& t, float amp)
{
	assert(t.backend() == t.backend());
	return Tensor(t.backend()->scalarMul(t.internalHandle(), amp), t.backend());
}


}
