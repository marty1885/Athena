#pragma once

#include <Athena/Backend/Backend.hpp>
#include <Athena/Utils/ReferenceCounter.hpp>
#include <Athena/Backend/TensorImpl.hpp>
#include <Athena/Utils/Error.hpp>
#include <Athena/Utils/BoxedValue.hpp>
#include <Athena/Utils/Type.hpp>
#include <Athena/DType.hpp>

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
	Tensor() = default;

	Tensor(TensorImpl* pimpl)
		: referenceCounter_(new ReferenceCounter(1)), pimpl_(pimpl) {}

	Tensor(const std::vector<float>& vec, const Shape& shape, Backend& backend = *defaultBackend())
		: Tensor(backend.createTensor(vec, shape)) {}

	Tensor(const std::vector<double>& vec, const Shape& shape, Backend& backend = *defaultBackend())
		: Tensor(backend.createTensor(vec, shape)) {}

	Tensor(const std::vector<int32_t>& vec, const Shape& shape, Backend& backend = *defaultBackend())
		: Tensor(backend.createTensor(vec, shape)) {}

	Tensor(const std::vector<int16_t>& vec, const Shape& shape, Backend& backend = *defaultBackend())
		: Tensor(backend.createTensor(vec, shape)) {}
	
	Tensor(const std::vector<bool>& vec, const Shape& shape, Backend& backend = *defaultBackend())
		: Tensor(backend.createTensor(vec, shape)) {}

	Tensor(nested_initializer_list_t<float, 1> l)
	{
		initWithIniter(l);
	}
	
	Tensor(nested_initializer_list_t<float, 2> l)
	{
		initWithIniter(l);
	}

	Tensor(nested_initializer_list_t<float, 3> l)
	{
		initWithIniter(l);
	}

	Tensor(nested_initializer_list_t<float, 4> l)
	{
		initWithIniter(l);
	}

	Tensor(nested_initializer_list_t<float, 5> l)
	{
		initWithIniter(l);
	}
	
	Tensor(nested_vector_t<float, 2> l)
	{
		initWithIniter(l);
	}

	Tensor(nested_vector_t<float, 3> l)
	{
		initWithIniter(l);
	}

	Tensor(nested_vector_t<float, 4> l)
	{
		initWithIniter(l);
	}

	Tensor(nested_vector_t<float, 5> l)
	{
		initWithIniter(l);
	}

	Tensor(const Tensor& t)
	{
		if(this == &t)
			return;

		pimpl_ = t.pimpl_;
		referenceCounter_ = t.referenceCounter_;
		if(referenceCounter_ != nullptr)
			referenceCounter_->addRef();
	}

	Tensor& operator= (const Tensor& other)
	{
		if(this == &other || other.pimpl() == nullptr)
			return *this;

		release();

		pimpl_ = other.pimpl_;
		referenceCounter_ = other.referenceCounter_;
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

	inline void reciprocate()
	{
		pimpl_->reciprocate();
	}

	Tensor slice(const Shape& begin, const Shape& size={1}) const
	{
		return pimpl_->slice(begin, size);
	}

	Tensor transpose() const
	{
		return pimpl_->transpose();
	}

	Tensor transpose(const std::vector<intmax_t>& axis) const
	{
		return pimpl_->transpose(axis);
	}

	Tensor clone() const
	{
		return Tensor(pimpl_->clone());
	}

	Tensor sum(intmax_t axis) const
	{
		return pimpl_->sum(axis);
	}

	Tensor sum(const std::vector<intmax_t>& axis) const
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

	Tensor reshape(const Shape& s) const
	{
		Tensor t = clone();
		if(s.contains(Shape::None))
			t.resize(solveUnknownDim(shape(), s));
		else
		{
			if(s.volume() != volume())
				throw AtError("Cannot reshape from " + to_string(shape()) + " to " + to_string(s));
			t.resize(s);
		}
		return t;
	}

	void resize(const Shape& s)
	{
		if(s.contains(Shape::None))
			pimpl_->resize(solveUnknownDim(shape(), s));
		else
		{
			if(s.volume() != volume())
				throw AtError("Cannot resize from " + to_string(shape()) + " to " + to_string(s));
			pimpl_->resize(s);
		}
	}

	Tensor greaterThan(float val) const
	{
		return pimpl_->greaterThan(val);
	}

	Tensor lesserThan(float val) const
	{
		return pimpl_->lesserThan(val);
	}

	Tensor greaterOrEqual(float val) const
	{
		return pimpl_->greaterOrEqual(val);
	}

	Tensor lesserOrEqual(float val) const
	{
		return pimpl_->lesserOrEqual(val);
	}

	Tensor equalTo(float val) const
	{
		return pimpl_->equalTo(val);
	}

	Tensor concatenate(const Tensor& other, intmax_t axis) const
	{
		return pimpl_->concatenate(other.pimpl(), axis);
	}

	Tensor exp() const
	{
		return pimpl_->exp();
	}

	Tensor log() const
	{
		return pimpl_->log();
	}

	size_t size() const
	{
		if((bool)(*this) == false)
			return 0;
		return pimpl_->size();
	}

	template <typename T>
	inline void host(T* ptr) const
	{
		pimpl_->host(ptr);
	}

	void flat()
	{
		resize({(intmax_t)size()});
	}

	Tensor flatten() const
	{
		Tensor t = std::move(clone());
		t.resize({(intmax_t)t.size()});
		return t;
	}

	template <typename T>
	std::vector<T> host() const
	{
		std::vector<T> v(pimpl_->size());
		pimpl_->host(&v[0]);
		return v;
	}

	intmax_t dimension() const
	{
		return shape().size();
	}

	intmax_t volume() const
	{
		return shape().volume();
	}

	Tensor withBackend(Backend& other)
	{
		return other.createTensor(host<float>(), shape());//Optimize this
	}

	template <typename T>
	inline T* hostPtr()
	{
		if(typeToDtype<T>() != dtype())
			throw AtError(std::string("Cannot get a ")  + to_string(typeToDtype<T>()) + " pointer from " + to_string(dtype()));
		return (T*)pimpl_->hostPtr();
	}

	template <typename T>
	inline const T* hostPtr() const
	{
		if(typeToDtype<T>() != dtype())
			throw AtError(std::string("Cannot get a ")  + to_string(typeToDtype<T>()) + " pointer from " + to_string(dtype()));
		return (const T*)pimpl_->hostPtr();
	}

	virtual ~Tensor()
	{
		release();
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

	static void setDefaultBackend(Backend* backend)
	{
		defaultBackend_ = backend;
	}

	static Backend* defaultBackend()
	{
		AtAssert(defaultBackend_ != nullptr, "Default backend not set. Please set it before using it.");
		return defaultBackend_;
	}

	explicit operator bool() const
	{
		return pimpl_ != nullptr;
	}

	BoxedValues states() const;

	void loadStates(const BoxedValues& states);

	void release()
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

	DType dtype() const
	{
		return pimpl_->dtype();
	}

	//Force evaulation since backends can do lazy evaulation
	void eval()
	{
		pimpl_->eval();
	}

	template <typename T> static Tensor from(const T& t);
	template <typename T> static T to(const Tensor& t);

protected:
	inline ReferenceCounter* referenceCounter() const
	{
		return referenceCounter_;
	}

	inline void setReferenceCounter(ReferenceCounter* ptr)
	{
		referenceCounter_ = ptr;
	}

	static Shape solveUnknownDim(const Shape in, const Shape& s)
	{
		int unknownCount = 0;
		intmax_t index = -1;
		intmax_t volume = 1;
		for(intmax_t i=0;i<(intmax_t)s.size();i++)
		{
			if(s[i] == Shape::None)
			{
				unknownCount++;
				index = i;
			}
			else
				volume *= s[i];
		}
		if(unknownCount == 1)
		{
			Shape res = s;
			if(in.volume()%volume != 0)
				throw AtError("Cannot solve unknow dimension for " + std::to_string(in.volume()) +" elements"
					+ " with shape " + to_string(s) + ". Cannot divide.");
			res[index] = in.volume()/volume;
			return res;
		}
		assert(unknownCount != 0);//Should not call this function with no unknowns
		throw AtError("Shape" + to_string(s) + " has more then 1 unknown dimentions. Cannot solve for unknown");
	}

	template<typename T>
	void initWithIniter(T l)
	{
		std::vector<float> data;
		Shape shape = shapeFromInitalizer(l);
		data.reserve(shape.volume());//Preallocate space for speed
		initlistToVector(l, data);
		//Dirty code
		*this = Tensor(data, shape, *defaultBackend());
	}

	template<typename T>
	static Shape shapeFromInitalizer(T l)
	{
		Shape s;
		return shapeFromInitalizerInternal(l ,s);
	}

	template<typename T>
	static Shape shapeFromInitalizerInternal(T l, Shape& s, size_t depth = 0)
	{
		if constexpr(!std::is_scalar<T>::value)
		{
			if(s.size() <= depth)
				s.push_back(l.size());
			else
			{
				if(s[depth] != (intmax_t)l.size())
					throw AtError("Un-uniform shape in initalizer. Expecting size of "
						+ std::to_string(s[depth]) + ", but get " + std::to_string(l.size()));
			}
			for(auto&& e : l)
				shapeFromInitalizerInternal(e, s, depth+1);
		}
		return s;
	}

	template<typename T>
	static void initlistToVector(T l, std::vector<float>& data)
	{
		if constexpr(!std::is_scalar<T>::value)
		{
			for(auto&& e : l)
				initlistToVector(e, data);
		}
		else
		{
			data.push_back(l);
		}
	}

	ReferenceCounter* referenceCounter_ = nullptr;
	TensorImpl* pimpl_ = nullptr;

	static Backend* defaultBackend_;
};

template <>
inline std::vector<bool> Tensor::host() const
{
	std::vector<bool> v(pimpl_->size());
	std::vector<char> c(pimpl_->size());
	pimpl_->host((bool*)&c[0]);
	for(size_t i=0;i<v.size();i++)
		v[i] = c[i];
	return v;
}

inline BoxedValues Tensor::states() const
{
	BoxedValues params;
	params.set<std::string>("__type", "Tensor");
	if(dtype() == DType::float32)
		params.set<std::vector<float>>("values", host<float>());
	else if(dtype() == DType::float64)
		params.set<std::vector<double>>("values", host<double>());
	else if(dtype() == DType::int32)
		params.set<std::vector<int32_t>>("values", host<int32_t>());
	else if(dtype() == DType::int16)
		params.set<std::vector<int16_t>>("values", host<int16_t>());
	else if(dtype() == DType::bool8)
		params.set<std::vector<bool>>("values", host<bool>());
	params.set<Shape>("shape", shape());
	return params;
}

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

inline Tensor ones(const Shape& shape, Backend& backend)
{
	return backend.ones(shape);
}


inline Tensor rand(float lEdge, float rEdge, const Shape& shape)
{
	return rand(lEdge, rEdge, shape, *Tensor::defaultBackend());
}

inline Tensor normal(float mean, float stddev, const Shape& shape)
{
	return normal(mean, stddev, shape, *Tensor::defaultBackend());
}

inline Tensor zeros(const Shape& shape)
{
	return zeros(shape, *Tensor::defaultBackend());
}

inline Tensor ones(const Shape& shape )
{
	return ones(shape, *Tensor::defaultBackend());
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

template <typename T>
static int prittyPrintTensor (std::ostream& os, T* arr, Shape shape, int depth, int maxDepth, int maxLength=0)
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

		int val = 0;
		for(intmax_t i=0;i<size;i++)
		{
			val = prittyPrintTensor(os, arr+i*vol, shape,depth+1, maxDepth, maxLength);
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
	if(t.pimpl() == nullptr)
	{
		os << "{}";
		return os;
	}

	if(t.dtype() == DType::float32)
	{
		auto v = t.host<float>();
		prittyPrintTensor(os, &v[0], t.shape(), 0, t.shape().size());
	}
	else if(t.dtype() == DType::float64)
	{
		auto v = t.host<double>();
		prittyPrintTensor(os, &v[0], t.shape(), 0, t.shape().size());
	}
	else if(t.dtype() == DType::int32)
	{
		auto v = t.host<int32_t>();
		prittyPrintTensor(os, &v[0], t.shape(), 0, t.shape().size());
	}
	else if(t.dtype() == DType::int16)
	{
		auto v = t.host<int16_t>();
		prittyPrintTensor(os, &v[0], t.shape(), 0, t.shape().size());
	}
	else if(t.dtype() == DType::bool8)
	{
		// auto v = t.host<bool>();
		// prittyPrintTensor(os, &v[0], t.shape(), 0, t.shape().size());
		throw AtError("Printing bool8 tensors not supported now");
	}
	else
		throw AtError("Cannot print tensor with type: " + to_string(t.dtype()));
	return os;
}

inline std::string to_string(const Tensor& t)
{
	std::stringstream ss;
	ss << t;
	return ss.str();
}

inline Tensor operator*(float val, const Tensor& t)
{
	Tensor res(t.clone());
	res.mul(val);
	return res;
}

inline Tensor operator-(const Tensor& t)
{
	return -1.f*t;
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
	Tensor res = -t;
	res.add(val);
	return res;
}

inline Tensor operator>(const Tensor& t, float val)
{
	return t.greaterThan(val);
}

inline Tensor operator<(const Tensor& t, float val)
{
	return t.lesserThan(val);
}

inline Tensor operator>=(const Tensor& t, float val)
{
	return t.greaterOrEqual(val);
}

inline Tensor operator<=(const Tensor& t, float val)
{
	return t.lesserOrEqual(val);
}

inline Tensor operator==(const Tensor& t, float val)
{
	return t.equalTo(val);
}

inline Tensor operator/(float amp, const Tensor& t)
{
	Tensor res(t.clone());
	res.reciprocate();
	if(amp != 1.f)
		res.mul(amp);
	return res;
}

inline Tensor exp(const Tensor& t)
{
	return t.exp();
}

inline Tensor log(const Tensor& t)
{
	return t.log();
}

}
