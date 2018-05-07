#pragma once

#include <string>
#include <type_traits>

#include <cstdint>

namespace At
{

enum class DType
{
	unknown = -1,
	float32 = 0,
	float64,
	int32,
	uint32,
	int16,
	uint16,
	int64,
	uint64,
	uint8,
	bool8
};

template <typename T>
inline DType typeToDtype()
{
	if(std::is_same<T, float>::value)
		return DType::float32;
	else if (std::is_same<T, double>::value)
		return DType::float64;
	else if (std::is_same<T, int32_t>::value)
		return DType::int32;
	else if (std::is_same<T, uint32_t>::value)
		return DType::uint32;
	else if (std::is_same<T, int16_t>::value)
		return DType::int16;
	else if (std::is_same<T, uint16_t>::value)
		return DType::uint16;
	else if (std::is_same<T, int64_t>::value)
		return DType::int64;
	else if (std::is_same<T, uint64_t>::value)
		return DType::uint64;
	else if (std::is_same<T, uint8_t>::value)
		return DType::uint8;
	else if (std::is_same<T, bool>::value)
		return DType::bool8;
	else
		return DType::unknown;
}

inline std::string dtypeToName(DType dtype)
{
	if(dtype == DType::float32)
		return "float32";
	else if(dtype == DType::float64)
		return "float64";
	else if(dtype == DType::int32)
		return "int32";
	else if(dtype == DType::uint32)
		return "uint32";
	else if(dtype == DType::int16)
		return "int16";
	else if(dtype == DType::uint16)
		return "uint16";
	else if(dtype == DType::int64)
		return "int64";
	else if(dtype == DType::uint64)
		return "uint64";
	else if(dtype == DType::uint8)
		return "uint8";
	else if(dtype == DType::bool8)
		return "bool8";
	else 
		return "unknown";
}

inline std::ostream& operator<< (std::ostream& os, const DType& dtype)
{
	os << dtypeToName(dtype);
	return os;
}

}