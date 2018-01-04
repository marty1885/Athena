#pragma once

#include <nnpack.h>

#include <Athena/Backend.hpp>

namespace At
{

class NNPackBackend : public Backend
{
public:
	virtual ~NNPackBackend();
	NNPackBackend(intmax_t threads = 1);

protected:
	pthreadpool_t threadpool_ = nullptr;
};

}
