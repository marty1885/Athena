#pragma once

#include <vector>

#include <Athena/Backend.hpp>
#include <Athena/Tensor.hpp>
#include <Athena/Optimizer.hpp>
#include <Athena/Utils/Shape.hpp>
#include <Athena/Layers/Layers.hpp>
#include <Athena/Loss.hpp>

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
	{
		backend_ = backend;
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
		if(input.shape()[0]%batchSize != 0)
			throw AtError("Error: batch size cannot divide the number of datasets perfectly.");

		if(input.shape()[0]<(intmax_t)batchSize)
			throw AtError("Error: batch size is larger than the number of datasets");

		size_t datasetSize = input.shape()[0];

		auto inputShape = input.shape();
		auto outputShape = desireOutput.shape();
		inputShape[0] = batchSize;
		outputShape[0] = batchSize;
		std::vector<Tensor> layerOutputs(layers_.size()+1);

		for(size_t i=0;i<epoch;i++)
		{
			float epochLoss = 0;
			for(size_t j=0;j<datasetSize;j+=batchSize)
			{
				Tensor x = input.slice({(intmax_t)j}, {(intmax_t)batchSize});
				Tensor y = desireOutput.slice({(intmax_t)j} ,{(intmax_t)batchSize});

				x.resize(inputShape);
				y.resize(outputShape);

				layerOutputs[0] = x.clone();

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
				Tensor l = loss.f(layerOutputs.back(), y);
				Tensor dE = E*l;

				for(int k=layers_.size()-1;k>=0;k--)
				{
					auto& layer = layers_[k];
					Tensor tmp;
					layer->backword(layerOutputs[k],layerOutputs[k+1], tmp, dE);
					if(layer->trainable())
						layer->update(&optimizer);

					dE = std::move(tmp);
				}
				float batchLoss = l.host()[0];
				onBatchEnumerate(batchLoss);
				epochLoss += batchLoss*(float)(batchSize)/datasetSize;
			}
			onEpochEnumerate(epochLoss);

		}
	}

	Tensor predict(const Tensor& input)
	{
		Tensor in = input.clone();
		for(auto& layer : layers_)
		{
			Tensor out = layer->forward(in);
			in = out;
		}

		return in;
	}

	const Layer* operator[](int index) const
	{
		return layers_[index];
	}

	Layer* operator[](int index)
	{
		return layers_[index];
	}

	size_t depth() const
	{
		return layers_.size();
	}

	Layer* getLayer(const std::string& name)
	{
		for(auto layer : layers_)
		{
			if(layer->name() == name)
				return layer;
		}
		return nullptr;
	}

	void compile();
	void summary(const Shape& inputShape) const;

protected:
	std::vector<Layer*> layers_;
	Backend* backend_;
};

}
