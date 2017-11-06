#include <Athena/Athena.hpp>

#include <iostream>
#include <chrono>

using namespace std::chrono;

int main()
{
	At::XtensorBackend backend;

	At::Tensor X({0,0, 1,0, 0,1, 1,1}, {4,2}, &backend);
	At::Tensor Y({0,1,1,0}, {4,1}, &backend);

	At::SequentialNetwork net;
	net.add<At::FullyConnectedLayer>(2,5, &backend);
	net.add<At::SigmoidLayer>(&backend);
	net.add<At::FullyConnectedLayer>(5,1, &backend);
	net.add<At::SigmoidLayer>(&backend);
	
	net.compile();

	net.summary();

	size_t epoch = 100000;

	At::NestrovOptimizer opt(&backend);
	At::MSELoss loss;
	opt.alpha_ = 0.35;
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	net.fit(opt,loss,X,Y,4,epoch);
	high_resolution_clock::time_point t2 = high_resolution_clock::now();

	duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
	std::cout << "It took me " << time_span.count() << " seconds." << std::endl;

	for(size_t i=0;i<X.shape()[0];i++)
	{
		At::Tensor x = X.slice({i},{1});
		At::Tensor res;
		net.predict(x, res);
		std::cout << "input = " << backend.get(x.internalHandle()) << ", result = " << backend.get(res.internalHandle()) << std::endl;
	}
}
