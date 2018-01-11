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

	intmax_t threads() const;

protected:
	pthreadpool_t threadpool_ = nullptr;
	intmax_t numThreads_ = 1;
};

}
