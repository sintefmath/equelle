/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#ifndef NODEINTERFACE_HEADER_INCLUDED
#define NODEINTERFACE_HEADER_INCLUDED

#include "EquelleType.hpp"
#include "FileLocation.hpp"

class ASTVisitorInterface;

/// Base class for all AST classes.
class Node
{
public:
    Node()
      :parent_(nullptr)
    {}
    virtual ~Node()
    {}
    virtual void accept(ASTVisitorInterface&)
    {
        // Do nothing.
    }
    FileLocation location() const
    {
        return loc_;
    }
    void setLocation(const FileLocation& loc)
    {
        loc_ = loc;
    }
    Node* getParent()
    {
        return parent_;
    }
    void setParent(Node* parent)
    {
        parent_ = parent;
    }
    virtual int numChildren() = 0;
    virtual Node* getChild(const int index) = 0;
    virtual void setChild(const int index, Node* child) = 0;
private:
    Node* parent_;
    // No copying.
    Node(const Node&);
    // No assignment.
    Node& operator=(const Node&);
    FileLocation loc_;
};


#endif // NODEINTERFACE_HEADER_INCLUDED
