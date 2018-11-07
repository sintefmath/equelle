#ifndef ASTREWRITER_HEADER_INCLUDED
#define ASTREWRITER_HEADER_INCLUDED

#include "ASTNodes.hpp"
class Node;

class ASTRewriter
{
public:
    void rewrite(Node* root, const int childIndex);
};

#endif // ASTREWRITER_HEADER_INCLUDED
