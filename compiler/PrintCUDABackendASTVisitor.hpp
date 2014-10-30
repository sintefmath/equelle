/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#ifndef PRINTCUDABACKENDASTVISITOR_HEADER_INCLUDED
#define PRINTCUDABACKENDASTVISITOR_HEADER_INCLUDED

#include "PrintCPUBackendASTVisitor.hpp"

class PrintCUDABackendASTVisitor : public PrintCPUBackendASTVisitor
{
public:
    PrintCUDABackendASTVisitor();
    ~PrintCUDABackendASTVisitor();

    // Overrides functions that need the name of CUDA backend class and namespace.
    const char* cppStartString() const;
    const char* cppEndString() const;
    const char* classNameString() const;
    const char* namespaceNameString() const;
};

#endif // PRINTCUDABACKENDASTVISITOR_HEADER_INCLUDED
