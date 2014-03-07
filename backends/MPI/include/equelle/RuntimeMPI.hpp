#pragma once

#include <memory>

class Zoltan;

namespace equelle {


/** RuntimeMPI is responsible for executing Equelle-simulators using MPI.
 *  It handles both the MPI context and the domain decomposition, using Zoltan. */
class  RuntimeMPI {
public:
    RuntimeMPI();
    virtual ~RuntimeMPI();

    //std::unique_ptr<Zoltan> zoltan;
};

}
