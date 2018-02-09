#include <Athena/Athena.hpp>
#include <Athena/XtensorBackend.hpp>

#include <iostream>
#include <chrono>

using namespace std::chrono;

int main()
{
	At::XtensorBackend backend;
	At::Tensor::setDefaultBackend(&backend);
	At::SequentialNetwork net(&backend);

	net.add(At::FullyConnectedLayer(2,5));
	net.add(At::SigmoidLayer());
	net.add(At::FullyConnectedLayer(5,1));
	net.add(At::SigmoidLayer());
	net.compile();

	net.summary({At::Shape::None, 2});

	At::Tensor X({0,0, 1,0, 0,1, 1,1}, {4,2});
	At::Tensor Y({0,1,1,0}, {4,1});

	At::NestrovOptimizer opt;
	At::MSELoss loss;
	opt.alpha_ = 0.35;

	size_t epoch = 100000;

	//Record how long it takes to train
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	net.fit(opt,loss,X,Y,4,epoch);
	high_resolution_clock::time_point t2 = high_resolution_clock::now();

	duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
	std::cout << "It took me " << time_span.count() << " seconds." << std::endl;

	for(int i=0;i<X.shape()[0];i++)
	{
		At::Tensor x = X.slice({i},{1});
		auto res = net.predict(x);
		std::cout << "input = " << x << ", result = " << res << std::endl;
	}
}
