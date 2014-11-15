
#include <vector>

#include "SeqOfScalar.hpp"


using namespace equelleCUDA;

SeqOfScalar equelleCUDA::operator*(const SeqOfScalar& seq, const Scalar a) {
    SeqOfScalar output = seq;
    for (int i = 0; i < seq.size(); i++) {
	output[i] *= a;
    }
    return output;
}
