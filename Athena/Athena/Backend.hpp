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

	virtual void* createTensor(const std::vector<float>& vec, const std::vector<size_t>& shape) = 0;
	virtual void* createTensor(const std::vector<size_t>& dims) = 0;
	virtual void* copyTensor(const void* src) = 0;
	virtual void destoryTensor(void* handle) = 0;

	virtual void device(void* handle, const float* ptr) = 0;
	virtual void host(void* handle, float* ptr) const = 0;
	virtual void* zeros(const std::vector<size_t>& shape) = 0;
	virtual void* ones(const std::vector<size_t>& shape) = 0;
	virtual void* rand(float lEdge, float rEdge, const std::vector<size_t>& shape) = 0;

	virtual void* add(const void* handle1,const  void* handle2) = 0;
	virtual void* multiply(const void* handle1,const  void* handle2) = 0;
	virtual void* scalarMul(const  void* handle, float x) = 0;
	virtual void* scalarAdd(const void* handle, float val) = 0;
	virtual void selfScalarAdd(void* handle, float val) = 0;
	virtual void* div(const void* handle1,const  void* handle2) = 0;
	virtual void* subtract(const void* handle1,const  void* handle2) = 0;

	virtual void* dot(const void* handle1, const void* handle2) = 0;

	virtual void* sum(const void* handle, const std::vector<size_t>& axis) = 0;

	virtual void* pow(const void* handle, float e) = 0;
	virtual void* sqrt(const void* handle) = 0;
	virtual void* abs(const void* handle) = 0;

	virtual std::vector<size_t> shape(void* handle) const = 0;
	virtual void reshape(void* handle, const std::vector<size_t>& targetShape) = 0;
	virtual void* transpose(void* handle) = 0;
	virtual void* slice(void* handle, const std::vector<size_t>& begin, const std::vector<size_t>& size) = 0;

	virtual size_t size(const void* handle) = 0;

	std::vector<float> host(void* handle)
	{
		size_t s = size(handle);
		std::vector<float> vec(s);
		host(handle, &vec[0]);
		return vec;
	}


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
	std::unordered_map<std::string, FunctoinWrapper*> algorithms_;
};


}
