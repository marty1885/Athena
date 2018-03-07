#pragma once

#include <vector>

#include <Athena/Backend.hpp>
#include <Athena/Tensor.hpp>
#include <Athena/Optimizer.hpp>
#include <Athena/Utils/Shape.hpp>
#include <Athena/Layers/Layers.hpp>
#include <Athena/Loss.hpp>

#include <type_traits>

namespace At
{

class SequentialNetwork
{
public:
	virtual ~SequentialNetwork()
	{
		for(auto layer : layers_)
			delete layer;
	}

	SequentialNetwork(Backend* backend)
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
		size_t batchSize, size_t epoch)
	{
		fit(optimizer, loss, input, desireOutput, batchSize, epoch, [](float){}, [](float){});
	}

	template <typename OnBatchEnumerate
		, typename OnEpochEnumerate>
	void fit(Optimizer& optimizer, LossFunction& loss, const Tensor& input, const Tensor& desireOutput,
		size_t batchSize, size_t epoch, OnBatchEnumerate onBatchEnumerate, OnEpochEnumerate onEpochEnumerate)
	{
		size_t datasetSize = input.shape()[0];

		auto inputShape = input.shape();
		auto outputShape = desireOutput.shape();
		std::vector<Tensor> layerOutputs(layers_.size()+1);

		for(size_t i=0;i<epoch;i++)
		{
			float epochLoss = 0;
			for(size_t j=0;j<datasetSize;j+=batchSize)
			{
				intmax_t sliceSize = std::min((intmax_t)batchSize, (intmax_t)(datasetSize-j));
				Tensor x = input.slice({(intmax_t)j}, {sliceSize});
				Tensor y = desireOutput.slice({(intmax_t)j} ,{sliceSize});

				inputShape[0] = sliceSize;
				outputShape[0] = sliceSize;

				x.resize(inputShape);
				y.resize(outputShape);

				//This variable should not be modified, no need to clone
				layerOutputs[0] = x;

				int index = 0;
				for(auto& layer : layers_)
				{
					const auto& currentInput = layerOutputs[index];
					Tensor out = layer->forward(currentInput);
					layerOutputs[++index] = std::move(out);
				}

				if(layerOutputs.back().shape() != y.shape())
					throw AtError("Expecting model output with shape " + to_string(y.shape())
						+ " but get " + to_string(layerOutputs.back().shape()));

				Tensor E = layerOutputs.back() - y;
				Tensor dE = loss.df(layerOutputs.back(), y);

				for(int k=layers_.size()-1;k>=0;k--)
				{
					auto& layer = layers_[k];
					Tensor tmp;
					layer->backword(layerOutputs[k],layerOutputs[k+1], tmp, dE);
					if(layer->trainable())
						layer->update(&optimizer);

					dE = std::move(tmp);
				}
				float batchLoss = loss.f(layerOutputs.back(), y).host()[0];
				onBatchEnumerate(batchLoss);
				epochLoss += batchLoss*((float)(sliceSize)/datasetSize);
			}
			onEpochEnumerate(epochLoss);

		}
	}

	Tensor predict(const Tensor& input)
	{
		Tensor t = input.clone();
		for(auto& layer : layers_)
		{
			Tensor out = layer->forward(t);
			t = out;
		}

		return t;
	}

	float test(const Tensor& input, const Tensor& desireOutput, LossFunction& loss)
	{
		Tensor t = input.clone();
		for(auto& layer : layers_)
		{
			Tensor out = layer->forward(t);
			t = out;
		}

		return loss.f(t, desireOutput).host()[0];
	}

	const Layer* operator[](int index) const
	{
		return layers_[index];
	}

	Layer* operator[](int index)
	{
		auto res = static_cast<const std::remove_reference<decltype(*this)>::type*>(this)->operator[] (index);
		return const_cast<Layer*>(res);
	}

	size_t depth() const
	{
		return layers_.size();
	}

	Layer* getLayer(const std::string& name)
	{
		auto it = std::find_if (layers_.begin(), layers_.end(), [&name](const auto& layer){return layer->name() == name;});
		if(it != layers_.end())
			return *it;
		return nullptr;
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
