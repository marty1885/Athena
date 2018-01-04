#pragma once

#include <nnpack.h>

#include <Athena/Backend.hpp>

namespace At
{

class NNPackBackend : public Backend
{
public:
	NNPackBackend();

protected:
	pthreadpool_t threadpool_ = nullptr;
};

}
