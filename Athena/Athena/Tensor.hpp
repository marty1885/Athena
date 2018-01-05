#pragma once

#include <Athena/Backend.hpp>
#include <Athena/ReferenceCounter.hpp>
#include <Athena/TensorImpl.hpp>

#include <assert.h>

#include <vector>
#include <numeric>
#include <iostream>
#include <sstream>

namespace At
{

class Tensor
{
public:
	Tensor()
	{
	}

	Tensor(const Shape& shape, Backend& backend)
		: Tensor(backend.createTensor(shape))
	{
	}

	Tensor(TensorImpl* pimpl)
		: referenceCounter_(new ReferenceCounter(1)), pimpl_(pimpl)
	{
	}

	Tensor(const std::vector<float>& vec, const Shape& shape, Backend& backend)
		: Tensor(backend.createTensor(vec, shape))
	{
	}

	Tensor(const Tensor& t)
	{
		if(this == &t)
			return;

		pimpl_ = (TensorImpl*)t.pimpl();
		referenceCounter_ = t.referenceCounter();
		if(referenceCounter_ != nullptr)
			referenceCounter_->addRef();
	}

	Tensor& operator= (const Tensor& other)
	{
		if(this == &other || other.pimpl() == nullptr)
			return *this;

		if(referenceCounter_ != nullptr && other.referenceCounter() != nullptr)
		{
			if(referenceCounter_->release() == 0)
			{
				delete referenceCounter_;
				backend()->destoryTensor(pimpl_);
			}
		}

		pimpl_ = (TensorImpl*)other.pimpl();
		referenceCounter_ = other.referenceCounter();
		if(referenceCounter_ != nullptr)
			referenceCounter_->addRef();

		return *this;
	}

	inline void add(float val)
	{
		pimpl_->add(val);
	}

	inline void mul(float val)
	{
		pimpl_->mul(val);
	}

	Tensor slice(const Shape& begin, const Shape& size={1}) const
	{
		return pimpl_->slice(begin, size);
	}

	Tensor transpose() const
	{
		return pimpl_->transpose();
	}

	Tensor clone() const
	{
		return Tensor(pimpl_->clone());
	}

	Tensor sum(intmax_t axis) const
	{
		return pimpl_->sum(axis);
	}

	Tensor pow(float e)
	{
		return pimpl_->pow(e);
	}

	Tensor dot(const Tensor& t) const
	{
		return pimpl_->dot(t.pimpl());
	}

	Tensor sqrt() const
	{
		return pimpl_->sqrt();
	}

	Tensor abs() const
	{
		return pimpl_->abs();
	}

	Tensor stack(const Tensor& t, intmax_t axis) const
	{
		return pimpl_->stack(t.pimpl(), axis);
	}

	const Shape shape() const
	{
		return pimpl_->shape();
	}

	void reshape(const Shape& s)
	{
		pimpl_->reshape(s);
	}

	Tensor concatenate(const Tensor& other, intmax_t axis) const
	{
		return pimpl_->concatenate(other.pimpl(), axis);
	}

	size_t size() const
	{
		return pimpl_->size();
	}

	void host(float* ptr) const
	{
		pimpl_->host(ptr);
	}

	void flat()
	{
		reshape({(intmax_t)size()});
	}

	Tensor flatten()
	{
		Tensor t = std::move(clone());
		t.reshape({(intmax_t)t.size()});
		return t;
	}

	std::vector<float> host() const
	{
		std::vector<float> v(pimpl_->size());
		pimpl_->host(&v[0]);
		return v;
	}

	//TODO: Implement this
	// Tensor transfer(Backend* otherBackend)
	// {
	// }

	float* hostPtr()
	{
		return pimpl_->hostPtr();
	}

	const float* hostPtr() const
	{
		return pimpl_->hostPtr();
	}

	virtual ~Tensor()
	{
		if(referenceCounter_ != nullptr)
		{
			if(referenceCounter_->release() == 0)
			{
				delete referenceCounter_;
				backend()->destoryTensor(pimpl_);
			}
		}
		referenceCounter_ = nullptr;
		pimpl_ = nullptr;
	}

	inline TensorImpl* pimpl()
	{
		return pimpl_;
	}

	inline const TensorImpl* pimpl() const
	{
		return pimpl_;
	}

	Backend* backend()
	{
		return pimpl_->backend();
	}

	Backend* backend() const
	{
		return pimpl_->backend();
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

	ReferenceCounter* referenceCounter_ = nullptr;
	TensorImpl* pimpl_ = nullptr;
};

inline Tensor rand(float lEdge, float rEdge, const Shape& shape, Backend& backend)
{
	return Tensor(backend.rand(lEdge, rEdge ,shape));
}

inline Tensor normal(float mean, float stddev, const Shape& shape, Backend& backend)
{
	return Tensor(backend.normal(mean, stddev, shape));
}

inline Tensor zeros(const Shape& shape, Backend& backend)
{
	return backend.zeros(shape);
}

inline Tensor dot(const Tensor& a, const Tensor& b)
{
	return Tensor(a.dot(b));
}

inline Tensor sqrt(const Tensor& t)
{
	return t.sqrt();
}

inline Tensor abs(const Tensor& t)
{
	return t.abs();
}

inline Tensor sum(const Tensor& t, intmax_t axis = 0)
{
	return t.sum(axis);
}

inline Tensor transpose(const Tensor& t)
{
	return t.transpose();
}

inline Tensor concatenate(const Tensor& t, const Tensor& q, intmax_t axis)
{
	return t.concatenate(q, axis);
}

inline Tensor stack(const Tensor& t, const Tensor& q, intmax_t axis)
{
	return t.stack(q, axis);
}

static int osTensorRecursive (std::ostream& os, float* arr, Shape shape, int depth, int maxDepth, int maxLength=0)
{
	auto floatToStr = [&](float val)->std::string
	{
		std::stringstream ss;
		ss << val;
		return ss.str();
	};
	if(shape.size() == 1)
	{
		os << "{ ";
		intmax_t size = shape[0];

		for(intmax_t i=0;i<size;i++)
		{
			std::string str = floatToStr(arr[i]);
			int len = maxLength-str.size();
			for(int i=0;i<len;i++)
				str += " ";
			os << str << (i==size-1 ? "" : ", ");
		}

		os << "}";
		return 1;
	}
	else
	{
		if(depth == 0)
		{
			for(int i=0;i<shape.volume();i++)
				maxLength = std::max(maxLength, (int)floatToStr(arr[i]).size());
		}

		intmax_t size = shape[0];
		shape.erase(shape.begin());
		intmax_t vol = shape.volume();

		os << "{";

		int val;
		for(intmax_t i=0;i<size;i++)
		{
			val = osTensorRecursive(os, arr+i*vol, shape,depth+1, maxDepth, maxLength);
			os << (i==size-1 ? "" : ", ");
			if(i != size-1)
			{
				for(int j=0;j<val;j++)
					os << '\n';
				for(int j=0;j<maxDepth-val;j++)
					os << ' ';
			}

		}
		os << "}";
		return val+1;
	}

}

inline std::ostream& operator<< (std::ostream& os, const Tensor& t)
{
	std::vector<float> v = t.host();
	osTensorRecursive(os, &v[0], t.shape(), 0, t.shape().size());
	return os;
}

inline Tensor operator+(const Tensor& t, const Tensor& other)
{
	//assert(t.backend() == t.backend());
	Tensor res(t.clone());
	res.pimpl()->add(other.pimpl());
	return res;
}

inline Tensor operator-(const Tensor& t, const Tensor& other)
{
	Tensor res(t.clone());
	res.pimpl()->subtract(other.pimpl());
	return res;
}

inline Tensor operator*(const Tensor& t, const Tensor& other)
{
	Tensor res(t.clone());
	res.pimpl()->mul(other.pimpl());
	return res;
}

inline Tensor operator/(const Tensor& t, const Tensor& other)
{
	Tensor res(t.clone());
	res.pimpl()->divide(other.pimpl());
	return res;
}

inline void operator-=(Tensor& t, const Tensor& other)
{
	t.pimpl()->subtract(other.pimpl());
}

inline void operator-=(Tensor& t, const float& x)
{
	t.add(-x);
}

inline void operator+=(Tensor& t, const Tensor& other)
{
	t.pimpl()->add(other.pimpl());
}

inline void operator+=(Tensor& t, const float& x)
{
	t.add(x);
}

inline void operator*=(Tensor& t, const Tensor& other)
{
	t.pimpl()->mul(other.pimpl());
}

inline void operator*=(Tensor& t, const float& x)
{
	t.mul(x);
}

inline void operator/=(Tensor& t, const Tensor& other)
{
	t.pimpl()->divide(other.pimpl());
}

inline void operator/=(Tensor& t, const float& x)
{
	t.mul(1.f/x);
}

inline Tensor operator+(const Tensor& t, float val)
{
	Tensor res(t.clone());
	res.add(val);
	return res;
}

inline Tensor operator-(const Tensor& t, float val)
{
	Tensor res(t.clone());
	res.add(-val);
	return res;
}

inline Tensor operator*(const Tensor& t, float amp)
{
	Tensor res(t.clone());
	res.mul(amp);
	return res;
}

inline Tensor operator/(const Tensor& t, float amp)
{
	Tensor res(t.clone());
	res.mul(1.f/amp);
	return res;
}

inline Tensor operator+(float val, const Tensor& t)
{
	Tensor res(t.clone());
	res.add(val);
	return res;
}

inline Tensor operator-(float val, const Tensor& t)
{
	Tensor res(t.clone());
	res.add(-val);
	return res;
}

inline Tensor operator*(float val, const Tensor& t)
{
	Tensor res(t.clone());
	res.mul(val);
	return res;
}

inline Tensor operator/(float amp, const Tensor& t)
{
	Tensor res(t.clone());
	res.mul(amp);
	return res;
}


}
