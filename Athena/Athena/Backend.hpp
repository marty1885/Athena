#pragma once

#include <vector>
#include <unordered_map>
#include <typeinfo>

#include <Athena/Error.hpp>
#include <Athena/Delegate.hpp>

namespace At
{
class Tensor;

template <typename FT>
using FuncType = FT;

using FCForwardFunction = FuncType<Tensor(const Tensor&,const Tensor&, const Tensor&)>;
using FCBackwardFunction = FuncType<Tensor(const Tensor&,const Tensor&)>;
using ActivationForward = FuncType<Tensor(const Tensor&)>;
using ActivationBackward = FuncType<Tensor(const Tensor&, const Tensor&)>;

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

	virtual size_t createTensor(const std::vector<float>& vec, const std::vector<size_t>& shape) = 0;
	virtual size_t createTensor(const std::vector<size_t>& dims) = 0;
	virtual size_t copyTensor(size_t src) = 0;
	virtual void destoryTensor(size_t handle) = 0;
	virtual size_t zeros(const std::vector<size_t>& shape) = 0;
	virtual size_t ones(const std::vector<size_t>& shape) = 0;
	virtual size_t rand(float lEdge, float rEdge, const std::vector<size_t>& shape) = 0;

	virtual size_t add(size_t handle1, size_t handle2) = 0;
	virtual size_t multiply(size_t handle1, size_t handle2) = 0;
	virtual size_t scalarMul(float x, size_t handle) = 0;
	virtual size_t scalarAdd(size_t handle, float val) = 0;
	virtual void selfScalarAdd(size_t handle, float val) = 0;
	virtual size_t div(size_t handle1, size_t handle2) = 0;
	virtual size_t subtract(size_t handle1, size_t handle2) = 0;

	virtual size_t dot(size_t handle1, size_t handle2) = 0;

	virtual std::vector<size_t> shape(size_t handle) const = 0;
	virtual void reshape(size_t handle, const std::vector<size_t>& targetShape) = 0;
	virtual size_t transpose(size_t handle) = 0;
	virtual size_t slice(size_t handle, const std::vector<size_t>& begin, const std::vector<size_t>& size) = 0;

	virtual size_t sum(size_t handle, const std::vector<size_t>& axis) = 0;

	virtual void device(size_t handle, const float* ptr) = 0;
	virtual void host(size_t handle, float* ptr) const = 0;

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
	std::vector<size_t> unusedSpace_;
	std::unordered_map<std::string, FunctoinWrapper*> algorithms_;
};


}
