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

	void host(float* ptr) const
	{
		backend_->host(handle_, ptr);
	}

	size_t size() const
	{
		return backend_->size(handle_);
	}

	void* internalHandle() const
	{
		return handle_;
	}

	void*& internalHandle()
	{
		return handle_;
	}

	Backend* backend() const
	{
		return backend_;
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
	ReferenceCounter* referenceCounter_ = nullptr;
	void* handle_ = nullptr;
};

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
