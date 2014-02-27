#include "equelle/equellecontroller.hpp"


class EquelleControllerImpl {
public:
    EquelleControllerImpl() {}
};

EquelleController
EquelleController::createEquelleController() {
    return EquelleController();
}

EquelleController::EquelleController() : pimpl( new EquelleControllerImpl() )
{
}

EquelleController::~EquelleController()
{
    delete pimpl;
}

