#pragma once

#include <vector>
#include <unordered_map>
#include <typeinfo>
#include <string>

#include <Athena/Error.hpp>
#include <Athena/Shape.hpp>
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

	virtual void* createTensor(const std::vector<float>& vec, const Shape& shape);
	virtual void* createTensor(const Shape& dims);
	virtual void* copyTensor(const void* src);
	virtual void destoryTensor(void* handle);

	virtual void device(void* handle, const float* ptr);
	virtual void host(void* handle, float* ptr) const;
	virtual void* zeros(const Shape& shape);
	virtual void* ones(const Shape& shape);
	virtual void* rand(float lEdge, float rEdge, const Shape& shape);
	virtual void* normal(float mean, float stddev, const Shape& shape);

	virtual void* add(const void* handle1,const  void* handle2);
	virtual void* multiply(const void* handle1,const  void* handle2);
	virtual void* scalarMul(const  void* handle, float x);
	virtual void* scalarAdd(const void* handle, float val);
	virtual void selfScalarAdd(void* handle, float val);
	virtual void* div(const void* handle1,const  void* handle2);
	virtual void* subtract(const void* handle1,const  void* handle2);

	virtual void* dot(const void* handle1, const void* handle2);

	virtual void* sum(const void* handle, const Shape& axis);

	virtual void* pow(const void* handle, float e);
	virtual void* sqrt(const void* handle);
	virtual void* abs(const void* handle);

	virtual Shape shape(void* handle) const;
	virtual void reshape(void* handle, const Shape& targetShape);
	virtual void* transpose(void* handle);
	virtual void* slice(void* handle, const Shape& begin, const Shape& size);
	virtual void* concatenate(const void* handle1, const void* handle2, int axis=0);
	virtual void* stack(const void* handle1, const void* handle2, int axis=0);

	virtual size_t size(const void* handle);

	inline std::string type() const
	{
		return type_;
	}

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

	void setType(const std::string& str)
	{
		type_ = str;
	}

	std::unordered_map<std::string, FunctoinWrapper*> algorithms_;
	std::string type_;
};


}
