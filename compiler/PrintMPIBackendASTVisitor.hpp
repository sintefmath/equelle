#pragma once

#include "PrintCPUBackendASTVisitor.hpp"

class PrintMPIBackendASTVisitor : public PrintCPUBackendASTVisitor
{
public:
    PrintMPIBackendASTVisitor();
    virtual ~PrintMPIBackendASTVisitor();

    const char* cppStartString() const;
    const char* cppEndString() const;
};

