#include <iostream>
#include <vector>

void generated_func()
{
    // --- start generated code (the rest is a generic template)
    std::vector<double> a = { 2, 4, 7, 1 };
    double b = a[0];
    std::vector<double> c = { 4, b };
    std::cout << c[1] << '\n';
    std::vector<double> d(a.size());
    for (int index = 0; index < a.size(); ++index) {
        double elem = a[index];
        d[index] = elem*elem;
    }
    std::cout << d[2] << '\n';
    // --- end generated code
}

int main()
{
    generated_func();
}

// Expected output start
// 2
// 49
// Expected output end
