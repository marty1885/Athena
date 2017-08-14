#include <Athena/Athena.hpp>

#include <iostream>

int main()
{
	xt::xarray<float> desireInput({{0,1},{1,1},{0,0,},{1,0}});
	xt::xarray<float> desireOutput({{0,1,0,0}});
	desireOutput = xt::transpose(desireOutput);

	//Feed forward neural network with size 2, 3, 1
	//The two weight matrix
	xt::xarray<float> syn0 = 2 * xt::random::rand<float>({2,3}) - 1;
	xt::xarray<float> syn1 = 2 * xt::random::rand<float>({3,1}) - 1;

	for(int i=0;i<10000;i++)
	{
		//Forward prpoergate
		for(unsigned int j=0;j<4;j++)
		{
			xt::xarray<float> l0 = xt::index_view(x, {{j,0},{j,1}});
			xt::xarray<float> l1 = activate(xt::linalg::dot(l0,syn0));
			xt::xarray<float> l2 = activate(xt::linalg::dot(l1,syn1));

			xt::xarray<float> target = y[j];

			xt::xarray<float> l2Error = target - l2;
                        //Uhh Then?
		}

	}

	std::cout << "Hello World" << '\n';
}
