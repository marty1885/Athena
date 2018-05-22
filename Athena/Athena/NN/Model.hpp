#pragma once

#include <vector>

#include <Athena/Backend/Backend.hpp>
#include <Athena/Tensor.hpp>
#include <Athena/NN/Optimizer.hpp>
#include <Athena/Utils/Shape.hpp>
#include <Athena/NN/Layers.hpp>
#include <Athena/NN/Loss.hpp>

#include <type_traits>

namespace At
{

class SequentialNetwork
{
public:
	virtual ~SequentialNetwork();

	SequentialNetwork(Backend* backend=Tensor::defaultBackend())
		: backend_(backend)
	{
	}

	SequentialNetwork(const SequentialNetwork&) = delete;
	SequentialNetwork(SequentialNetwork&& other)
	{
		layers_ = other.layers_;
		backend_ = other.backend_;

		other.layers_.clear();
	}

	template<typename LayerType, typename ... Args>
	void add(Args ... args)
	{
		layers_.push_back(new LayerType(args ...));
	}

	template<typename LayerType>
	void add(LayerType layer)
	{
		layers_.push_back(new LayerType(layer));
	}

	template<typename LayerType>
	SequentialNetwork& operator <<(LayerType layer)
	{
		layers_.push_back(new LayerType(layer));
		return *this;
	}

	void fit(Optimizer& optimizer, LossFunction& loss, const Tensor& input, const Tensor& desireOutput,
		size_t batchSize, size_t epoch);

	void fit(Optimizer& optimizer, LossFunction& loss, const Tensor& input, const Tensor& desireOutput,
		size_t batchSize, size_t epoch, delegate<void(float)> onBatchEnumerate, delegate<void(float)> onEpochEnumerate);

	Tensor predict(const Tensor& input);

	float test(const Tensor& input, const Tensor& desireOutput, LossFunction& loss);

	inline const Layer* operator[](int index) const
	{
		return layers_[index];
	}

	inline Layer* operator[](int index)
	{
		auto res = static_cast<const std::remove_reference<decltype(*this)>::type*>(this)->operator[] (index);
		return const_cast<Layer*>(res);
	}

	inline size_t depth() const
	{
		return layers_.size();
	}

	template <typename T=Layer>
	T* layer(const std::string& name)
	{
		return const_cast<T*>(static_cast<const SequentialNetwork*>(this)->layer<T>(name));
	}

	template <typename T=Layer>
	const T* layer(const std::string& name) const
	{
		auto it = std::find_if (layers_.begin(), layers_.end(), [&name](const auto& layer){return layer->name() == name;});
		if(it != layers_.end())
		{
			T* ptr = dynamic_cast<T*>(*it);
			if(ptr == nullptr)
				throw AtError("Layer " + name + " cannot be casted to " + typeid(T).name());
			return ptr;
		}
		return nullptr;
	}

	template <typename T=Layer>
	T* layer(int i)
	{
		return const_cast<T*>(static_cast<const SequentialNetwork*>(this)->layer(i));
	}

	template <typename T=Layer>
	const T* layer(int i) const
	{
		T* ptr = dynamic_cast<T*>(layers_[i]);
		if(ptr == nullptr)
			throw AtError("Layer " + std::to_string(i) + " in network cannot be casted to " + typeid(T).name());
		return ptr;
	}

	void compile();
	void summary(const Shape& inputShape) const;
	virtual BoxedValues states() const;
	virtual void loadStates(const BoxedValues& states);
	void save(const std::string path) const;
	void load(const std::string path);

protected:
	std::vector<Layer*> layers_;
	Backend* backend_;
};

template<typename ActivationType>
inline SequentialNetwork makeMLP(std::vector<intmax_t> layerSize, Backend& backend)
{
	SequentialNetwork net(&backend);
	if(layerSize.size() < 2)
		throw AtError("A MLP must have at least 2 layers, got " + std::to_string(layerSize.size()));
	for(size_t i=1;i<layerSize.size();i++)
		net << FullyConnectedLayer(layerSize[i-1], layerSize[i]) << ActivationType();
	net.compile();
	return std::move(net);
}

}
