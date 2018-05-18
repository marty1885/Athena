#include <Athena/Utils/Archive.hpp>
#include <Athena/Utils/Error.hpp>
#include <Athena/Utils/Shape.hpp>

#include <msgpack.hpp>

#include <fstream>
using namespace At;

template <typename Stream, typename T>
void packChild(msgpack::packer<Stream>& o, std::string type, const T& val)
{
	o.pack_map(1);
	o.pack(type);
	o.pack(val);
}

template <typename T, typename VT>
inline T unpackChild(VT const& o)
{
	T v;
	o.convert(v);
	return v;
}

namespace msgpack {
MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS) {
namespace adaptor {

template<>
struct pack<BoxedValues> {
	template <typename Stream>
	packer<Stream>& operator()(msgpack::packer<Stream>& o, BoxedValues const& states) const {
		// packing member variables as a map
		o.pack_map(states.size());
		for(const auto& [key, elem] : states)
		{
			o.pack(key);
			if(auto ptr = box_cast<BoxedValues>(elem); ptr != nullptr)
				o.pack(ptr->value());
			else if(auto ptr = box_cast<std::string>(elem); ptr != nullptr)
				o.pack(ptr->value());
			else if(auto ptr = box_cast<Shape>(elem); ptr != nullptr)
				packChild(o, "Shape", ptr->value());
			else if(auto ptr = box_cast<std::vector<float>>(elem); ptr != nullptr)
				packChild(o, "FloatVector", ptr->value());
			else if(auto ptr = box_cast<std::vector<int32_t>>(elem); ptr != nullptr)
				packChild(o, "Int32Vector", ptr->value());
			else if(auto ptr = box_cast<std::vector<int16_t>>(elem); ptr != nullptr)
				packChild(o, "Int16Vector", ptr->value());
			else if(auto ptr = box_cast<std::vector<double>>(elem); ptr != nullptr)
				packChild(o, "Float64Vector", ptr->value());
			else if(auto ptr = box_cast<std::vector<bool>>(elem); ptr != nullptr)
				packChild(o, "BoolVector", ptr->value());
			else if(auto ptr = box_cast<float>(elem); ptr != nullptr)
				packChild(o, "Float32", ptr->value());
			else if(auto ptr = box_cast<int>(elem); ptr != nullptr)
				packChild(o, "Int32", ptr->value());
			else
				throw AtError("Not supported type");
		}
		return o;
	}
};

template<>
struct convert<BoxedValues> {
	msgpack::object const& operator()(msgpack::object const& o, BoxedValues& states) const {
		if (o.type != msgpack::type::MAP) throw msgpack::type_error();
		const msgpack::object_map& m = o.via.map;
		auto size = m.size;
		msgpack::object_kv* kvs = m.ptr;
		for(size_t i=0;i<size;i++)
		{
			std::string key;
			kvs[i].key.convert(key);
			const auto& val = kvs[i].val;
			if(val.type == msgpack::type::STR)
				states.set(key, unpackChild<std::string>(val));
			else if(val.type == msgpack::type::MAP)
			{
				if(val.via.map.size == 1)
				{
					const auto& kv = *val.via.map.ptr;
					std::string type = kv.key.as<std::string>();
					if(type == "FloatVector")
						states.set(key, unpackChild<std::vector<float>>(kv.val));
					if(type == "Float64Vector")
						states.set(key, unpackChild<std::vector<double>>(kv.val));
					else if(type == "Int32Vector")
						states.set(key, unpackChild<std::vector<int32_t>>(kv.val));
					else if(type == "Int16Vector")
						states.set(key, unpackChild<std::vector<int16_t>>(kv.val));
					else if(type == "BoolVector")
						states.set(key, unpackChild<std::vector<bool>>(kv.val));
					else if(type == "Shape")
						states.set(key, unpackChild<Shape>(kv.val));
					else if(type == "Float32")
						states.set(key, unpackChild<float>(kv.val));
					else if(type == "Int32")
						states.set(key, unpackChild<int>(kv.val));
				}
				else
					states.set(key, unpackChild<BoxedValues>(val));
			}
			else
				states.set(key, unpackChild<BoxedValues>(val));
		}
		return o;
	}
};

template<>
struct pack<Shape> {
	template <typename Stream>
	packer<Stream>& operator()(msgpack::packer<Stream>& o, Shape const& s) const {
		// packing member variables as an array.
		o.pack_array(s.size());
		for(auto v : s)
			o.pack(v);
		
		return o;
	}
};

template<>
struct convert<Shape> {
	msgpack::object const& operator()(msgpack::object const& o, Shape& s) const {
		if (o.type != msgpack::type::ARRAY) throw msgpack::type_error();
		auto size = o.via.array.size;
		s.resize(size);
		for(size_t i=0;i<size;i++)
			s[i] = o.via.array.ptr[i].as<Shape::value_type>();
		return o;
	}
};

} // namespace adaptor
} // MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
} // namespace msgpack


void Archiver::save(const BoxedValues& states, std::string path)
{
	std::ofstream out(path);
	if(out.good() == false)
		throw AtError("Cannot save to " + path);
	msgpack::pack(out, states);
	out.close();
}


BoxedValues Archiver::load(std::string path)
{
	BoxedValues vals;
	std::ifstream in(path);
	if(in.good() == false)
		throw AtError("Cannot load file : " + path);
	std::stringstream buffer;
	buffer << in.rdbuf();
	msgpack::object_handle oh = msgpack::unpack(buffer.str().data(), buffer.str().size());
	msgpack::object obj = oh.get();
	obj.convert(vals);
	return vals;
}
