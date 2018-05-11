#include <Athena/Tensor.hpp>
#include <Athena/Backend/Backend.hpp>

using namespace At;
Backend* Tensor::defaultBackend_ = nullptr;