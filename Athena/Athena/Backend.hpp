#pragma once

#include <vector>
#include <unordered_map>
#include <typeinfo>
#include <string>

#include <Athena/Error.hpp>
#include <Athena/Shape.hpp>
#include <Athena/Delegate.hpp>
#include <Athena/TensorImpl.hpp>

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

struct FunctoinWrapper
{
	virtual ~FunctoinWrapper(){}
};

template<typename FT>
struct FunctionContainer : public FunctoinWrapper
{
	delegate<FT> func;
	FunctionContainer(delegate<FT> f) : func(std::move(f))
	{
	}

	inline delegate<FT> get() const
	{
		return func;
	}
};

class Backend
{
public:
	virtual ~Backend()
	{
		for(auto& it : algorithms_)
			delete it.second;
	}

	virtual TensorImpl* createTensor(const std::vector<float>& vec, const Shape& shape);
	virtual TensorImpl* createTensor(const Shape& dims);
	virtual void destoryTensor(TensorImpl* handle);

	virtual TensorImpl* zeros(const Shape& shape);
	virtual TensorImpl* ones(const Shape& shape);
	virtual TensorImpl* rand(float lEdge, float rEdge, const Shape& shape);
	virtual TensorImpl* normal(float mean, float stddev, const Shape& shape);

	template<typename FT>
	inline void addAlgorithm(const std::string& name, delegate<FT> f)
	{
		algorithms_[name] = new FunctionContainer<FT>(f);
	}

	template<typename FT>
	delegate<FT> getAlgorithm(const std::string& name, const std::vector<int64_t> params = {})
	{
		auto it = algorithms_.find(name);
		if(it != algorithms_.end())
		{
			//Remove RTTI if it turns out to be too slow
			FunctionContainer<FT>* container = dynamic_cast<FunctionContainer<FT>*>(it->second);
			if(container != nullptr)
				return container->get();
			throw AtError("Algorithm \"" + name + "\" is not typed as \"" + typeid(FT).name());
		}
		throw AtError("Algorithm \"" + name + "\" not found");
	}

protected:

	void setType(const std::string& str)
	{
		type_ = str;
	}

	std::unordered_map<std::string, FunctoinWrapper*> algorithms_;
	std::string type_;
};


}
