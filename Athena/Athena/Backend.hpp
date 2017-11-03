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

	virtual void* createTensor(const std::vector<float>& vec, const std::vector<size_t>& shape) = 0;
	virtual void* createTensor(const std::vector<size_t>& dims) = 0;
	virtual void* copyTensor(const void* src) = 0;
	virtual void destoryTensor(void* handle) = 0;

	virtual void device(void* handle, const float* ptr) = 0;
	virtual void host(void* handle, float* ptr) const = 0;
	virtual void* zeros(const std::vector<size_t>& shape) = 0;
	virtual void* ones(const std::vector<size_t>& shape) = 0;
	virtual void* rand(float lEdge, float rEdge, const std::vector<size_t>& shape) = 0;

	virtual void* add(void* handle1, void* handle2) = 0;
	virtual void* multiply(void* handle1, void* handle2) = 0;
	virtual void* scalarMul(float x, void* handle) = 0;
	virtual void* scalarAdd(void* handle, float val) = 0;
	virtual void selfScalarAdd(void* handle, float val) = 0;
	virtual void* div(void* handle1, void* handle2) = 0;
	virtual void* subtract(void* handle1, void* handle2) = 0;

	virtual void* dot(void* handle1, void* handle2) = 0;

	virtual void* sum(void* handle, const std::vector<size_t>& axis) = 0;
	virtual void* pow(void* handle, float e) = 0;

	virtual std::vector<size_t> shape(void* handle) const = 0;
	virtual void reshape(void* handle, const std::vector<size_t>& targetShape) = 0;
	virtual void* transpose(void* handle) = 0;
	virtual void* slice(void* handle, const std::vector<size_t>& begin, const std::vector<size_t>& size) = 0;


	virtual size_t size(void* handle) = 0;

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
