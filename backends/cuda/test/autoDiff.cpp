#include <iostream>
#include <vector>

#include "cos_soon.hpp"
#include "CudaArray.hpp"

using namespace equelleCUDA;


int main() {
    
    int size = 100;

    std::vector<double> a_vec ,b_vec;
    for (int i = 0; i < size; ++i) {
	a_vec.push_back((i*2 + 3) * (1/7));
	b_vec.push_back(i);
    }
    CollOfScalarSoon a(a_vec);
    CollOfScalarSoon b(b_vec);
    
    std::cout << "a.size(): " << a.size() << std::endl;
    std::vector<double> b_back = b.copyToHost();
    std::cout << "b[13] = " << b_back[13] << std::endl;

    CollOfScalarSoon c = a+b;
    std::vector<double> c_back = c.copyToHost();
    for (int i = 0; i < size; i+=10) {
	std::cout << "c_back = " << c_back[i] << " =? " << a_vec[i] + b_vec[i]
		  << " a_vec + b_vec\n";
    }


    CudaArray a_ca(a_vec);
    CudaArray b_ca(b_vec);
    CudaArray c_ca = a_ca + b_ca;

    //CudaArray a_ca = a.val();
    //CudaArray b_ca = b.val();

    return 0;
}
