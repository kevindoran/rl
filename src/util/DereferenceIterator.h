#ifndef REINFORCEMENT_DEREFERENCEITERATOR_H
#define REINFORCEMENT_DEREFERENCEITERATOR_H

// Inspired from:
// https://jonasdevlieghere.com/containers-of-unique-pointers/

// See the boost version. It is more feature rich, but much harder to read:
// https://github.com/boostorg/iterator/blob/develop/include/boost/iterator/indirect_iterator.hpp

#include <cstddef> // For std::size_t.

namespace rl {
namespace util {

template <typename BaseIterator>
class DereferenceIterator : public BaseIterator {
public:
    using Value = typename BaseIterator::value_type::element_type;
    using Pointer = Value*;
    using Reference = Value&;

    DereferenceIterator(const BaseIterator& other) : BaseIterator(other) {}

    Reference operator*() const {
        return *(this->BaseIterator::operator*());
    }

    Pointer operator->() const {
        return this->BaseIterator::operator*().get();
    }

    Reference operator[](std::size_t n) const {
        return *(this->BaseIterator::operator[](n));
    }
};

} // namespace util
} // namespace rl

#endif //REINFORCEMENT_DEREFERENCEITERATOR_H
