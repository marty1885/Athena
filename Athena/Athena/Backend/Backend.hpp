#pragma once

#include <vector>
#include <unordered_map>
#include <typeinfo>
#include <string>
#include <array>

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
	virtual ~Backend()
	{
		for(auto& it : algorithms_)
		{
			for(auto& ptr : it.second)
				delete ptr;
		}
	}

	virtual TensorImpl* createTensor(const std::vector<float>& vec, const Shape& shape);
	virtual TensorImpl* createTensor(const Shape& dims);
	virtual void destoryTensor(TensorImpl* handle);

	virtual TensorImpl* zeros(const Shape& shape);
	virtual TensorImpl* ones(const Shape& shape);
	virtual TensorImpl* rand(float lEdge, float rEdge, const Shape& shape);
	virtual TensorImpl* normal(float mean, float stddev, const Shape& shape);

	template<typename FT>
	inline void addAlgorithm(const std::string& name, delegate<FT> f, AlgorithmSelector selector = [](const BoxedValues& config)->bool{return true;})
	{
		algorithms_[name].push_back(new FunctionContainer<FT>(f, selector));
	}

	template<typename FT>
	FunctionContainer<FT> getFunction(const std::string name) const
	{
		auto it = algorithms_.find(name);
		if(it != algorithms_.end())
		{
			auto ptr = dynamic_cast<FunctionContainer<FT>*>(it->second.back());
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
			//Remove RTTI if it turns out to be too slow
			for(auto& algo : backwards(it->second))
			{
				FunctionContainer<FT>* container = dynamic_cast<FunctionContainer<FT>*>(algo);
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

protected:

	void setType(const std::string& str)
	{
		type_ = str;
	}

	std::unordered_map<std::string, std::vector<FunctoinWrapper*>> algorithms_;
	std::string type_;
};


}
