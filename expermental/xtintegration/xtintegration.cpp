#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>

#include <Athena/Backend/XtensorBackend.hpp>
#include <Athena/Tensor.hpp>
#include <Athena/Athena.hpp>
#include <Athena/XtensorIntegration.hpp>

int main()
{
	At::XtensorBackend backend;
	At::Tensor::setDefaultBackend(&backend);
	xt::xarray<float> arr = xt::zeros<float>({5,5});
	At::Tensor t = At::Tensor::from(arr);
	std::cout << t << std::endl;

	std::cout << At::Tensor::to<xt::xarray<float>>(t) << std::endl;
}
