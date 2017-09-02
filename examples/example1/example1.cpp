#include <Athena/Athena.hpp>

#include <iostream>
#include <vector>
#include <algorithm>

int main()
{
	At::SequentialNetwork net;
	net.add<At::FullyConnectedLayer>(2,5);
	net.add<At::SigmoidLayer>();
	net.add<At::FullyConnectedLayer>(5,1);
	net.add<At::SigmoidLayer>();

	xt::xarray<float> X = {{0,0},{1,0},{0,1},{1,1}};
	xt::xarray<float> Y = {{0,1,1,0}};
	Y = xt::transpose(Y);

	int epoch = 1000;

	At::SGDOptimizer opt;
	net.fit(opt,X,Y,2,epoch);

	for(int i=0;i<(int)X.shape()[0];i++)
	{
		xt::xarray<float> x = xt::view(X,i);
		xt::xarray<float> res;
		net.predict(x, res);
		std::cout << "input = " << x << ", result = " << res << std::endl;
	}
}
