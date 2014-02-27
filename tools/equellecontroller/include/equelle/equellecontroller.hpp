#pragma once

/**
EquelleController is a C++03 interface to generated EquelleSimulators that
allow for simulator setup, execution and monitoring from other applications.
The interface is made to be easily SWIGable to allow for Equelle-interoperation from
other languages, such as Python and Java.

Since EquelleController is designed to be passed across compile boundaries it does not
use any part of STL.

EquelleController is a very small class and can safely be stored in containers directly.
*/

class EquelleControllerImpl; // Forward declaration.

/* EquelleController uses the pimpl-idiom. */
class EquelleController {
public:
    /** Create an instance of an EquelleController that controls a given simulator. */
    static EquelleController createEquelleController();

    ~EquelleController();
private:
    EquelleController();
    EquelleControllerImpl* pimpl;
};


