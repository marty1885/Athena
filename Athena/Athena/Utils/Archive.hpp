#pragma once

#include <Athena/Utils/BoxedValue.hpp>
#include <Athena/Utils/Shape.hpp>
#include <Athena/Utils/Error.hpp>

#include <string>
#include <iostream>

#include <msgpack.hpp>

namespace At
{

class Archiver
{
public:

	static void save(const BoxedValues& states, std::string path);
	static BoxedValues load(std::string path);
};


}
