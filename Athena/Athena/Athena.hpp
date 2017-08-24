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
	Layer(int input, int output):
		mInputShape({input}), mOutputShape({output})
	{
	}

	virtual void forward(const xt::xarray<float>& input, xt::xarray<float>& output)
	{
	}

	virtual void backword(const xt::xarray<float>& x, xt::xarray<float>& y,
			xt::xarray<float>& dx, const xt::xarray<float>& dy)
	{
	}

// protected:
	std::vector<xt::xarray<float>> mWeights;
	std::vector<int> mInputShape;
	std::vector<int> mOutputShape;
};

class FullyConnected : public Layer
{
public:
	FullyConnected(int input, int output):
		Layer(input, output)
	{
		mWeights.push_back(2 * xt::random::rand<float>({input, output}) - 1);
		mWeights.push_back(2 * xt::random::rand<float>({output}) - 1);
	}

	virtual void forward(const xt::xarray<float>& x, xt::xarray<float>& y) override
	{
		y = xt::linalg::dot(x, mWeights[0])+mWeights[1];
	}

	virtual void backword(const xt::xarray<float>& x, xt::xarray<float>& y,
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
		y = 1/(1+xt::exp(x));
	}

	virtual void backword(const xt::xarray<float>& x, xt::xarray<float>& y,
			xt::xarray<float>& dx, const xt::xarray<float>& dy) override
	{
		dx = dy * (y * (1 - y));
	}
};

class SequentialNetwork
{
public:
	inline xt::xarray<float> activate(const xt::xarray<float>& x, bool diriv = false)
	{
		if(diriv)
			return x*(1-x);
		return 1/(1+xt::exp(-x));
	}

	template<typename LayerType, typename ... Args>
	void add(Args ... args)
	{
		mLayers.push_back(new LayerType(args ...));
	}

	void fit(const xt::xarray<float>& input, const xt::xarray<float>& desireOutput,
		int epoch)
	{
		int datasetSize = input.shape()[0];
		float L = 0.45f;
		for(int i=0;i<epoch;i++)
		{
			for(int j=0;j<datasetSize;j++)
			{
				xt::xarray<float> x = xt::view(input,j,xt::all(),xt::all());
				xt::xarray<float> y = xt::view(desireOutput,j,xt::all(),xt::all());

				//TODO: handle N-D data
				x.reshape({1,x.shape()[0]});
				y.reshape({1,y.shape()[0]});

				std::vector<xt::xarray<float>> layerOutputs;
				layerOutputs.push_back(x);

				for(auto& layer : mLayers)
				{
					auto& currentInput = layerOutputs.back();
					xt::xarray<float> out;
					layer->forward(currentInput, out);
					out = activate(out);
					layerOutputs.push_back(out);
				}

				xt::xarray<float> E = y - layerOutputs.back();
				xt::xarray<float> dE = E * activate(layerOutputs.back(), true);

				for(int k=mLayers.size()-1;k>=0;k--)
				{
					auto& layer = mLayers[k];
					xt::xarray<float> tmp;
					layer->backword(layerOutputs[k],layerOutputs[k+1], tmp, dE);
					tmp *= activate(layerOutputs[k], true);
					layer->mWeights[0] += xt::linalg::dot(xt::transpose(layerOutputs[k]), dE)*L;
					layer->mWeights[1] += dE*L;

					dE = tmp;
				}

				std::cout << E << std::endl;
			}

		}
	}
protected:
	std::vector<Layer*> mLayers;
};

}
