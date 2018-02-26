
#include "ASTRewriter.hpp"
#include "ASTNodes.hpp"
#include <iostream>
#include <stdexcept>
#include <sstream>


void ASTRewriter::rewrite(Node* root, const int childIndex)
{
    
    if(root == nullptr) { return; }


    switch (typeid(root)) {
        case typeid(SequenceNode*):
            SequenceNode* current = dynamic_cast<SequenceNode*>(right_);
            int i = 0;
            for( auto n : current->nodes() )
            {
                n->setParent(current);
                rewrite(n, i);
                i += i;
            }
            break;
        case typeid(BinaryOpNode*):
            
            BinaryOpNode* current = dynamic_cast<BinaryOpNode*>(root);
            auto leftChild = dynamic_cast<BinaryOpNode*>(current->getChild(0));
            auto rightChild = dynamic_cast<BinaryOpNode*>(current->getChild(1));
            leftChild->setParent(current);
            rightChild->setParent(current);

            rewrite(leftChild, 0);
            rewrite(rightChild, 1);

            if ( current->op() == Add ) {

                // a + b * c
                if ( rightChild != nullptr && rightChild->op() == Multiply ) {
                    auto replacementNode = new MultiplyAddNode(leftChild,rightChild->getChild(0), rightChild->getChild(1))
                    replaceNode(childIndex, current, replacementNode)
                }
                
                // a * b + c
                if ( leftChild == nullptr && leftChild->op() == Multiply ) {
                    auto replacementNode = new MultiplyAddNode(rightChild,leftChild->getChild(0), leftChild->getChild(1))
                    replaceNode(childIndex, current, replacementNode)
                }

            }

        default:
            int numChildren = root->numChildren();

            for ( size_t i = 0; i < numChildren; i++ ) {
                root->getChild(i)->setParent(root);
                rewrite(root->getChild(i), i);
            }
            break;
    }
}

void replaceNode(const int childIndex, Node* currentNode, Node* replacementNode)
{
    currentNode->setChild(0,nullptr);
    currentNode->setChild(1,nullptr);
    currentNode->getParent()->setChild(childIndex,replacementNode);
    delete current;
}