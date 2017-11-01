#pragma once

#include <Athena/Backend.hpp>

#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xindexview.hpp>
#include <xtensor/xvectorize.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include <assert.h>

#include <vector>

namespace At
{

class Tensor
{
public:
	Tensor()
	{
	}

	Tensor(const std::vector<size_t>& shape, Backend* backend)
	{
		backend_ = backend;
		handle_ = backend->createTensor(shape);
	}
	
	Tensor(size_t handle, Backend* backend)
	{
		handle_ = handle;
		backend_ = backend;
	}

	Tensor(const std::vector<float>& vec, const std::vector<size_t>& shape, Backend* backend)
	{
		
		backend_ = backend;
		handle_ = backend->createTensor(vec, shape);
	}

	Tensor(const Tensor& t)
	{
		//Workarround someone trying to copy a not initialized Tensor
		if(t.backend() == nullptr)
			return;
		backend_ = t.backend();
		handle_ = backend_->copyTensor(t.internalHandle());
	}
	
	Tensor& operator= (const Tensor& other)
	{
		if(other.backend() == nullptr)
			return *this;
		if(handle_ != 0)
			backend_->destoryTensor(handle_);
		backend_ = other.backend();
		handle_ = backend_->copyTensor(other.internalHandle());
		return *this;
	}

	Tensor(Tensor&& t)
	{
		backend_ = t.backend();
		handle_ = t.internalHandle();
		t.internalHandle() = 0;
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

	const std::vector<size_t> shape() const
	{
		return backend_->shape(handle_);
	}
	
	void operator-=(const Tensor& other)
	{
		//Optimize this
		*this = (*this - other);
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
		if(handle_ != 0)
			backend_->destoryTensor(handle_);
		handle_ = 0;
		backend_ = nullptr;
	}

protected:
	Backend* backend_ = nullptr;
	size_t handle_ = 0;
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
