#pragma once

#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xindexview.hpp>
#include <xtensor/xvectorize.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include <assert.h>

#include <vector>
#include <iostream>

namespace At
{

class Layer
{
public:
	Layer()
	{
	}
	Layer(int input, int output):
		mInputShape({input}), mOutputShape({output})
	{
	}

	virtual void forward(const xt::xarray<float>& input, xt::xarray<float>& output)
	{
	}

	virtual void backword(const xt::xarray<float>& x, const xt::xarray<float>& y,
			xt::xarray<float>& dx, const xt::xarray<float>& dy)
	{
	}

// protected:
	std::vector<xt::xarray<float>> mWeights;
	std::vector<int> mInputShape;
	std::vector<int> mOutputShape;
};

class Optimizer
{
public:
	virtual void update(xt::xarray<float>& weights, const xt::xarray<float>& grads) = 0;
};

class SGDOptimizer
{
public:
	SGDOptimizer(float alpha = 0.45) : mAlpha(alpha)
	{
	}
	virtual void update(xt::xarray<float>& weights, const xt::xarray<float>& grads)
	{
		weights += grads*mAlpha;
	}

	float mAlpha = 0.45f;
};

class FullyConnectedLayer : public Layer
{
public:
	FullyConnectedLayer(int input, int output):
		Layer(input, output)
	{
		mWeights.push_back(2 * xt::random::rand<float>({input, output}) - 1);
		mWeights.push_back(2 * xt::random::rand<float>({output}) - 1);
	}

	virtual void forward(const xt::xarray<float>& x, xt::xarray<float>& y) override
	{
		y = xt::linalg::dot(x, mWeights[0])+mWeights[1];
	}

	virtual void backword(const xt::xarray<float>& x, const xt::xarray<float>& y,
			xt::xarray<float>& dx, const xt::xarray<float>& dy) override
	{
		dx = xt::linalg::dot(dy,xt::transpose(mWeights[0]));
	}
};

class SigmoidLayer : public Layer
{
public:
	virtual void forward(const xt::xarray<float>& x, xt::xarray<float>& y) override
	{
		y = 1/(1+xt::exp(-x));
	}

	virtual void backword(const xt::xarray<float>& x, const xt::xarray<float>& y,
			xt::xarray<float>& dx, const xt::xarray<float>& dy) override
	{
		dx = dy * (y * (1 - y));
	}
};

class TanhLayer : public Layer
{
public:
	virtual void forward(const xt::xarray<float>& x, xt::xarray<float>& y) override
	{
		y = xt::tanh(x);
	}

	virtual void backword(const xt::xarray<float>& x, const xt::xarray<float>& y,
			xt::xarray<float>& dx, const xt::xarray<float>& dy) override
	{
		dx = dy * (1 - xt::pow(xt::tanh(y), 2));
	}
};

class ReluLayer : public Layer
{
public:
	virtual void forward(const xt::xarray<float>& x, xt::xarray<float>& y) override
	{
		y = xt::vectorize([](float v){return v > 0 ? v : 0.f;})(x);
	}

	virtual void backword(const xt::xarray<float>& x, const xt::xarray<float>& y,
			xt::xarray<float>& dx, const xt::xarray<float>& dy) override
	{
		dx = dy * xt::vectorize([](float v){return v > 0 ? 1.f : 0.f;})(y);
	}
};

class SequentialNetwork
{
public:
	template<typename LayerType, typename ... Args>
	void add(Args ... args)
	{
		mLayers.push_back(new LayerType(args ...));
	}

	void fit(Optimizer& optimizer, const xt::xarray<float>& input, const xt::xarray<float>& desireOutput,
		int batchSize, int epoch)
	{
		assert(input.shape()[0]%batchSize == 0);
		int datasetSize = input.shape()[0];

		auto inputShape = input.shape();
		auto outputShape = desireOutput.shape();
		inputShape[0] = batchSize;
		outputShape[0] = batchSize;

		std::vector<float> epochLoss(datasetSize);

		for(int i=0;i<epoch;i++)
		{
			for(int j=0;j<datasetSize;j++)
			{
				xt::xarray<float> x = xt::view(input,xt::range(j,j+batchSize));
				xt::xarray<float> y = xt::view(desireOutput,xt::range(j,j+batchSize));

				x.reshape(inputShape);
				y.reshape(outputShape);

				std::vector<xt::xarray<float>> layerOutputs;
				layerOutputs.push_back(x);

				for(auto& layer : mLayers)
				{
					auto& currentInput = layerOutputs.back();
					xt::xarray<float> out;
					layer->forward(currentInput, out);
					layerOutputs.push_back(out);
				}

				xt::xarray<float> E = y - layerOutputs.back();
				xt::xarray<float> dE = E;

				for(int k=mLayers.size()-1;k>=0;k--)
				{
					auto& layer = mLayers[k];
					xt::xarray<float> tmp;
					layer->backword(layerOutputs[k],layerOutputs[k+1], tmp, dE);
					if(layer->mWeights.size() > 0)
					{
						optimizer.update(layer->mWeights[0], xt::linalg::dot(xt::transpose(layerOutputs[k]), dE));
						optimizer.update(layer->mWeights[1], xt::sum(dE,{0})*learningRate);
					}

					dE = tmp;
				}
				epochLoss[j] = ((xt::xarray<float>)xt::sum(xt::pow(E,2)))[0];
			}
			std::cout << std::accumulate(epochLoss.begin(), epochLoss.end(), 0.f)
				/epochLoss.size() << std::endl;
		}
	}

	void predict(const xt::xarray<float>& input, xt::xarray<float>& output)
	{
		std::vector<xt::xarray<float>> layerOutputs;
		layerOutputs.push_back(input);

		for(auto& layer : mLayers)
		{
			auto& currentInput = layerOutputs.back();
			xt::xarray<float> out;
			layer->forward(currentInput, out);
			layerOutputs.push_back(out);
		}

		output = layerOutputs.back();
	}

	Layer* operator[](int index)
	{
		return mLayers[index];
	}

	std::vector<Layer*> mLayers;
	float learningRate = 0.45;
};

}
