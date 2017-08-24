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

	xt::xarray<float> X = {{0,0},{1,0},{0,1},{1,1}};
	xt::xarray<float> Y = {{0,1,1,0}};
	Y = xt::transpose(Y);

	int epoch = 3000;

	At::SequentialNetwork net;
	net.add<At::FullyConnected>(2,5);
	net.add<At::FullyConnected>(5,1);
	net.fit(X,Y,epoch);
}
