#include "ASTRewriter.hpp"
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <typeinfo>

void replaceNode(const int childIndex, Node* currentNode, Node* replacementNode)
{
    currentNode->setChild(0,nullptr);
    currentNode->setChild(1,nullptr);
    currentNode->getParent()->setChild(childIndex,replacementNode);
    delete currentNode;
}

void ASTRewriter::rewrite(Node* root, const int childIndex)
{
    

    // The rewriter traverses and modifies the subtree of the node root in a pre-order fashion.
    // This means that the algorithm first recurses to the bottom of the tree and modifies the
    // tree from the bottom and up.
    if(root == nullptr) { return; }

    if(typeid(root) == typeid(SequenceNode*)) {
        SequenceNode* current = dynamic_cast<SequenceNode*>(root);
        int i = 0;
        for( auto n : current->nodes() ){
            n->setParent(current);
            rewrite(n, i);
            i += i;
        }
    }else
    if(typeid(root) == typeid(BinaryOpNode*)) {
        
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
                auto replacementNode = 
                        new MultiplyAddNode(leftChild, 
                            dynamic_cast<ExpressionNode*>(rightChild->getChild(0)), 
                            dynamic_cast<ExpressionNode*>(rightChild->getChild(1)));
                replaceNode(childIndex, current, replacementNode);
            }
            
            // a * b + c
            if ( leftChild == nullptr && leftChild->op() == Multiply ) {
                auto replacementNode = 
                        new MultiplyAddNode(rightChild,
                            dynamic_cast<ExpressionNode*>(leftChild->getChild(0)), 
                            dynamic_cast<ExpressionNode*>(leftChild->getChild(1)));
                replaceNode(childIndex, current, replacementNode);
            }

        }
    }else{   
        int numChildren = root->numChildren();
        for ( size_t i = 0; i < numChildren; i++ ) {
            root->getChild(i)->setParent(root);
            rewrite(root->getChild(i), i);
        }
    }
}
/*
void replaceNode(const int childIndex, Node* currentNode, Node* replacementNode)
{
    currentNode->setChild(0,nullptr);
    currentNode->setChild(1,nullptr);
    currentNode->getParent()->setChild(childIndex,replacementNode);
    delete current;
}*/