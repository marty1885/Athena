#pragma once

#include <assert.h>

#include <vector>
#include <iostream>
#include <array>
#include <unordered_map>
#include <map>
#include <string>
#include <sstream>

#include <Athena/Backend.hpp>
#include <Athena/Tensor.hpp>
#include <Athena/Optimizer.hpp>
#include <Athena/Shape.hpp>
#include <Athena/Layers/Layers.hpp>

namespace At
{

class LossFunction
{
public:
	virtual Tensor f(const Tensor& y, const Tensor& t) = 0;

	virtual Tensor df(const Tensor& y, const Tensor& t)
	{
		return Tensor();
	}
};

class MSELoss : public LossFunction
{
	virtual Tensor f(const Tensor& y, const Tensor& t) override
	{
		return (y-t).pow(2.f).sum(0)/(float)y.shape()[0];
	}

	virtual Tensor df(const Tensor& y, const Tensor& t) override
	{
		float factor = 2.f/(float)t.size();
		return factor*(y - t);
	}
};

using L2Loss = MSELoss;

class AbsoluteLoss : public LossFunction
{
	virtual Tensor f(const Tensor& y, const Tensor& t)
	{
		return sum(abs(y-t));
	}

	// virtual void df(const Tensor& y, const Tensor& t, Tensor& d) override
	// {
	// 	d.reshape(t.shape());
	// 	float factor = 1.f/(float)t.size();
	// 	auto func = [factor](float x)->float{return x < 0.f? -factor : (x > 0.f ? factor : 0.f);};
        //
	// 	d = xt::vectorize(func)(y-t);
	// }
};

using L1Loss = AbsoluteLoss;

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

	void compile()
	{
		std::map<std::string, int> layerTypeNum;

		for(auto& layer : layers_)
		{
			if(layer->name() == "")
			{
				auto layerType = layer->type();
				//std::map initializes variable by default even  when accessing it
				layer->setName(layerType + "_" + std::to_string(++layerTypeNum[layerType]));
			}

			if(layer->backend() == nullptr)
				layer->setBackend(backend_);

			layer->build();
		}
	}

	void summary() const
	{
		auto repeat = [](const std::string& str, int n) {std::ostringstream os;for(int i=0;i<n;i++)os << str; return os.str();};
		auto trimString = [](const std::string& str, size_t n){
			std::string res;
			res.reserve(n);
			size_t end = std::min(str.size(), n);
			res += str.substr(0, end);
			if(end == n)
				return res;
			return res+std::string(n-end, ' ');
		};

		std::cout << repeat("─",80) << '\n';
		std::cout << trimString("Layer (type)",23) << " " << trimString("Output shape", 15) << " " << trimString("Params #", 16) << '\n';
		size_t trainableWeights = 0;
		for(size_t i=0;i<depth();i++)
		{
			if(i == 0)
				std::cout << repeat("=",80) << '\n';
			else
				std::cout << repeat("─",80) << '\n';
			const auto& l = layers_[i];
			std::cout << trimString(l->name()+" ("+l->type()+")", 23) << " ";

			const auto& shape = l->outputShape();
			std::ostringstream stream;
			stream << shape;
			std::string str =  stream.str();
			std::cout << trimString(str, 15) << " ";

			if(l->trainable() == false)
				std::cout << trimString("0", 16);
			else
			{
				size_t val = 0;
				const auto& weights = l->weights();
				for(const auto& w : weights)
					val += w.size();
				trainableWeights += val;
				std::cout << trimString(std::to_string(val), 15) << " ";
			}

			std::cout << '\n';
			if(i != depth()-1)
				std::cout << '\n';
		}

		std::cout << repeat("=",80) << '\n';

		std::cout << "Trainable weights: " << trainableWeights << '\n';
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

				x.reshape(inputShape);
				y.reshape(outputShape);

				layerOutputs[0] = x.clone();

				int index = 0;
				for(auto& layer : layers_)
				{
					const auto& currentInput = layerOutputs[index];
					Tensor out = layer->forward(currentInput);
					layerOutputs[++index] = std::move(out);
				}

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

	std::vector<Layer*> layers_;
	Backend* backend_;
};

}
