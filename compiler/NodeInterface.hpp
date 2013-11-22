/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#ifndef NODEINTERFACE_HEADER_INCLUDED
#define NODEINTERFACE_HEADER_INCLUDED

#include "EquelleType.hpp"

class ASTVisitorInterface;

/// Base class for all AST classes.
class Node
{
public:
    Node()
    {}
    virtual ~Node()
    {}
    virtual EquelleType type() const
    {
        return EquelleType();
    }
    virtual void accept(ASTVisitorInterface&)
    {
        // Do nothing.
    }
private:
    // No copying.
    Node(const Node&);
    // No assignment.
    Node& operator=(const Node&);
};


#endif // NODEINTERFACE_HEADER_INCLUDED
