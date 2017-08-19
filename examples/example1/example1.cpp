#include <Athena/Athena.hpp>


#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

inline xt::xarray<float> activate(const xt::xarray<float>& x, bool diriv = false)
{
	if(diriv)
		return x*(1-x);
	return 1/(1+xt::exp(-x));
};

int main()
{
	unsigned int epoch = 3000;
	int inputLayerSize = 2;
	int hiddenLayerSize = 5;
	int outputLayerSize = 1;
	float learningRate = 0.45;

	xt::xarray<float> Wh = 2.0f * xt::random::rand<float>({inputLayerSize,hiddenLayerSize}) - 1;
	xt::xarray<float> Wz = 2.0f * xt::random::rand<float>({hiddenLayerSize,outputLayerSize}) - 1;
	xt::xarray<float> Bh = 2.0f * xt::random::rand<float>({hiddenLayerSize}) - 1;
	xt::xarray<float> Bz = 2.0f * xt::random::rand<float>({outputLayerSize}) - 1;

	xt::xarray<float> X = {{1,0},{0,1},{1,1},{0,0}};
	unsigned int datasetSize = X.shape()[0];

	//Workarround xtensor issue #389 (https://github.com/QuantStack/xtensor/issues/389)
	//TODO: Remove this workarround after bug in xtensor fixed
	xt::xarray<float> tmp = {{1,1,0,0}};
	auto Y = xt::transpose(tmp);

	for(unsigned int i=0;i<epoch;i++)
	{
		float lossSum = 0;
		for(unsigned int j=0;j<datasetSize;j++)
		{
			xt::xarray<float> Xb = xt::view(X,j,xt::all(),xt::all());
			xt::xarray<float> Yb = xt::index_view(Y,{{0,j}});

			xt::xarray<float> H = activate(xt::linalg::dot(Xb, Wh)+Bh);
			xt::xarray<float> Z = activate(xt::linalg::dot(H, Wz)+Bz);
			xt::xarray<float> E = Yb - Z;

			xt::xarray<float> dZ = E * activate(Z, true);
			xt::xarray<float> dH = xt::linalg::dot(dZ, xt::transpose(Wz)) * activate(H, true);
			Wz += xt::linalg::dot(xt::transpose(H), dZ)*learningRate;
			Wh += xt::linalg::dot(xt::transpose(Xb), dH)*learningRate;
			Bz += dZ*learningRate;
			Bh += dH*learningRate;

			lossSum += ((xt::xarray<float>)xt::sum(xt::pow(E,2)))[0];
		}

		cout << lossSum/datasetSize << endl;
	}
}
