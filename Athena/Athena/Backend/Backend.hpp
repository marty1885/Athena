#pragma once

#include <vector>
#include <unordered_map>
#include <typeinfo>
#include <string>
#include <array>
#include <memory>

#include <Athena/Utils/Error.hpp>
#include <Athena/Utils/Shape.hpp>
#include <Athena/Utils/Delegate.hpp>
#include <Athena/Utils/BoxedValue.hpp>
#include <Athena/Backend/TensorImpl.hpp>

namespace At
{
class Tensor;

template <typename FT>
using FuncType = FT;

using FCForwardFunction = FuncType<Tensor(const Tensor&,const Tensor&, const Tensor&)>;
using FCBackwardFunction = FuncType<Tensor(const Tensor&,const Tensor&)>;
using SigmoidForward = FuncType<Tensor(const Tensor&)>;
using SigmoidBackward = FuncType<Tensor(const Tensor&, const Tensor&)>;
using TanhForward = FuncType<Tensor(const Tensor&)>;
using TanhBackward = FuncType<Tensor(const Tensor&, const Tensor&)>;
using ReluForward = FuncType<Tensor(const Tensor&)>;
using ReluBackward = FuncType<Tensor(const Tensor&, const Tensor&)>;
using Conv2DForward = FuncType<Tensor(const Tensor&, const Tensor&, const Tensor&, const Shape&)>;
using Conv2DBackward = FuncType<Tensor(const Tensor& , const Tensor& , Tensor& , Tensor&, const Tensor&, const Shape&)>;
using LeakyReluForward = FuncType<Tensor(const Tensor& x, float alpha)>;
using LeakyReluBackward = FuncType<Tensor(const Tensor& a, const Tensor& b, float alpha)>;

using AlgorithmSelector = delegate<bool(const BoxedValues& config)>;

struct FunctoinWrapper
{
	AlgorithmSelector selector_;
	FunctoinWrapper(AlgorithmSelector selector) :
		selector_(selector)
	{
	}

	FunctoinWrapper() :
		selector_([](const BoxedValues& config)->bool{return true;})
	{
	}

	bool sutiable(const BoxedValues& config)
	{
		return selector_(config);
	}

	AlgorithmSelector selector() const
	{
		return selector_;
	}

	virtual ~FunctoinWrapper() = default;

};

template <typename T>
class iterate_backwards
{
public:
	explicit iterate_backwards(const T &t) : t(t) {}
	typename T::const_reverse_iterator begin() const { return t.rbegin(); }
	typename T::const_reverse_iterator end()   const { return t.rend(); }
private:
	const T &t;
};
template <typename T>
iterate_backwards<T> backwards(const T &t)
{
	return iterate_backwards<T>(t);
}

template<typename FT>
struct FunctionContainer : public FunctoinWrapper
{
	delegate<FT> func_;
	FunctionContainer(delegate<FT> f) : func_(std::move(f))
	{
	}

	FunctionContainer(delegate<FT> f, AlgorithmSelector selector)
		: FunctoinWrapper(selector), func_(std::move(f))
	{
	}

	inline delegate<FT> get() const
	{
		return func_;
	}

	explicit operator bool() const
	{
		return (bool)func_;
	}

	//Unfortunatelly making a function-call operator is impossible
	template <typename ResType, typename ... Args>
	inline ResType call(Args&& ... args)
	{
		  return func_(args ...);
	}
};

template <typename RetType, typename ... Args>
RetType invoke(FunctoinWrapper& wrapper, Args&& ... args)
{
	using FT = RetType(Args ...);
	auto container = dynamic_cast<FunctionContainer<FT>*>(&wrapper);
	AtAssert(container != nullptr, std::string("Cannot cast Function to ") + typeid(FT).name());
	return container->get()(args ...);
}

class Backend
{
public:
	virtual ~Backend() = default;

	virtual TensorImpl* createTensor(const std::vector<float>& vec, const Shape& shape);
	virtual TensorImpl* createTensor(const std::vector<double>& vec, const Shape& shape);
	virtual TensorImpl* createTensor(const std::vector<int32_t>& vec, const Shape& shape);
	virtual TensorImpl* createTensor(const std::vector<int16_t>& vec, const Shape& shape);
	virtual TensorImpl* createTensor(const std::vector<bool>& vec, const Shape& shape);
	virtual TensorImpl* createTensor(const Shape& dims);
	virtual TensorImpl* clone(const TensorImpl* handle);
	virtual void destoryTensor(TensorImpl* handle);

	virtual TensorImpl* zeros(const Shape& shape, DType dtype=DType::float32);
	virtual TensorImpl* ones(const Shape& shape, DType dtype=DType::float32);
	virtual TensorImpl* rand(float lEdge, float rEdge, const Shape& shape);
	virtual TensorImpl* normal(float mean, float stddev, const Shape& shape);

	virtual void eval(TensorImpl* impl);

	virtual Shape shape(const TensorImpl* impl) const;
	virtual intmax_t size(const TensorImpl* impl) const;
	virtual DType dtype(const TensorImpl* impl) const;

	virtual void selfReciprocate(TensorImpl* impl);
	virtual void selfAdd(TensorImpl* impl, float val);
	virtual void selfMul(TensorImpl* impl, float val);
	virtual void selfAdd(TensorImpl* impl, const TensorImpl* other);
	virtual void selfMul(TensorImpl* impl, const TensorImpl* other);
	virtual void selfSub(TensorImpl* impl, const TensorImpl* other);
	virtual void selfDiv(TensorImpl* impl, const TensorImpl* other);
	virtual TensorImpl* add(const TensorImpl* impl, const TensorImpl* other);
	virtual TensorImpl* mul(const TensorImpl* impl, const TensorImpl* other);
	virtual TensorImpl* sub(const TensorImpl* impl, const TensorImpl* other);
	virtual TensorImpl* div(const TensorImpl* impl, const TensorImpl* other);

	virtual TensorImpl* sqrt(const TensorImpl* impl);
	virtual TensorImpl* abs(const TensorImpl* impl);
	virtual TensorImpl* exp(const TensorImpl* impl);
	virtual TensorImpl* log(const TensorImpl* impl);
	virtual TensorImpl* pow(const TensorImpl* impl, float val);

	virtual TensorImpl* dot(const TensorImpl* impl, const TensorImpl* other);

	virtual void modDims(TensorImpl* impl, const Shape& wantedShape);
	virtual TensorImpl* reshape(const TensorImpl* impl, const Shape& wantedShape);
	virtual TensorImpl* transpose(const TensorImpl* impl);

	virtual TensorImpl* sum(const TensorImpl* impl, intmax_t axis);
	virtual TensorImpl* sum(const TensorImpl* impl, const std::vector<intmax_t>& axis);
	virtual TensorImpl* stack(const TensorImpl* impl, const TensorImpl* other, int axis);
	virtual TensorImpl* concatenate(const std::vector<TensorImpl const*>& arrs, int axis);
	virtual TensorImpl* chunk(const TensorImpl* impl, const Shape& begin, const Shape& size);

	virtual void host(const TensorImpl* impl, float* ptr) const;
	virtual void host(const TensorImpl* impl, double* ptr) const;
	virtual void host(const TensorImpl* impl, int32_t* ptr) const;
	virtual void host(const TensorImpl* impl, int16_t* ptr) const;
	virtual void host(const TensorImpl* impl, bool* ptr) const;

	virtual void device(TensorImpl* impl, const float* ptr);
	virtual void device(TensorImpl* impl, const double* ptr);
	virtual void device(TensorImpl* impl, const int32_t* ptr);
	virtual void device(TensorImpl* impl, const int16_t* ptr);
	virtual void device(TensorImpl* impl, const bool* ptr);

	virtual void* hostPtr(TensorImpl* impl);
	virtual const void* hostPtr(const TensorImpl* impl);

	virtual TensorImpl* greaterThan(const TensorImpl* impl,float val);
	virtual TensorImpl* lesserThan(const TensorImpl* impl,float val);
	virtual TensorImpl* greaterOrEqual(const TensorImpl* impl,float val);
	virtual TensorImpl* lesserOrEqual(const TensorImpl* impl,float val);
	virtual TensorImpl* equalTo(const TensorImpl* impl,float val);

	template<typename FT>
	inline void addAlgorithm(const std::string& name, delegate<FT> f, AlgorithmSelector selector = [](const BoxedValues& config)->bool{return true;})
	{
		algorithms_[name].push_back(std::make_shared<FunctionContainer<FT>>(f, selector));
	}

	template<typename FT>
	FunctionContainer<FT> getFunction(const std::string name) const
	{
		auto it = algorithms_.find(name);
		if(it != algorithms_.end())
		{
			auto ptr = dynamic_cast<FunctionContainer<FT>*>(it->second.back().get());
			return *ptr;
		}
		throw AtError("Cannot find algorithm " + name + ".");
	}

	template<typename FT>
	delegate<FT> getAlgorithm(const std::string& name, const BoxedValues& config = BoxedValues(), bool checkConditions = true) const
	{
		auto it = algorithms_.find(name);
		if(it != algorithms_.end())
		{
			for(auto& algo : backwards(it->second))
			{
				FunctionContainer<FT>* container = dynamic_cast<FunctionContainer<FT>*>(algo.get());
				bool good = true;
				if(checkConditions == true && config.size() != 0)
					good = algo->sutiable(config);
				if(container != nullptr && good == true)
					return container->get();
					
			}
		}
		return delegate<FT>();
	}

	std::string type() const
	{
		return type_;
	}

	template <typename FT, typename BT>
	void useAlgorithm(const std::string& name, const BT& other)
	{
		if(&other == this)//For good measure
			return;
		auto algo = other.template getFunction<FT>(name);
		if((bool)algo == false)
			throw AtError(std::string("Algorithm ") + name + "with type " + typeid(FT).name() + " cannot be found ");
		addAlgorithm<FT>(name, algo.get(), algo.selector());
	}

	const std::unordered_map<std::string, std::vector<std::shared_ptr<FunctoinWrapper>>>& algorithms() const
	{
		return algorithms_;
	}

	void useAllAlgorithm(const Backend& other)
	{
		if(&other == this)
			return;
		for(auto [name, algos] : algorithms_)
		{
			auto& vec = algorithms_[name];
			vec.reserve(vec.size() + algos.size());
			vec.insert(vec.end(), algos.begin(), algos.end());
		}
	}

protected:

	void setType(const std::string& str)
	{
		type_ = str;
	}

	std::unordered_map<std::string, std::vector<std::shared_ptr<FunctoinWrapper>>> algorithms_;
	std::string type_;
};


}
