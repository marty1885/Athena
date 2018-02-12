#pragma once

#include <type_traits>

namespace At
{
template<typename Test, template<typename...> class Ref>
struct is_specialization : std::false_type {};

template<template<typename...> class Ref, typename... Args>
struct is_specialization<Ref<Args...>, Ref>: std::true_type {};

template <class T, std::size_t I>
struct nested_initializer_list
{
	using type = std::initializer_list<typename nested_initializer_list<T, I - 1>::type>;
};

template <class T>
struct nested_initializer_list<T, 0>
{
	using type = T;
};

template <class T, std::size_t I>
using nested_initializer_list_t = typename nested_initializer_list<T, I>::type;

template <class T, std::size_t I>
struct nested_vector
{
	using type = std::vector<typename nested_vector<T, I - 1>::type>;
};

template <class T>
struct nested_vector<T, 0>
{
	using type = T;
};

template <class T, std::size_t I>
using nested_vector_t = typename nested_vector<T, I>::type;


}