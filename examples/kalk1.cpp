#include <iostream>

void generated_func()
{
    // --- start generated code (the rest is a generic template)
    double a = 1;
    double b = 2;
    double c = double(a + 4*b)/double(3 - a);   // Note: x/y -> double(x)/double(y)
    double d = double(1)/double(2);
    std::cout << c << ' ' << d << '\n';
    // --- end generated code
}

int main()
{
    generated_func();
}

// Expected output start
// 4.0 0.5
// Expected output end
