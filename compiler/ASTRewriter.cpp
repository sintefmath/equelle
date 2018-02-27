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
    

    // The rewriter traverses and modifies the subtree of the root node in a pre-order fashion.
    // This means that the algorithm first recurses to the bottom of the tree and modifies the
    // tree from the bottom and up.
    if(root == nullptr) { return; }

    if(typeid(*root) == typeid(SequenceNode*)) {
        SequenceNode* current = dynamic_cast<SequenceNode*>(root);
        int i = 0;
        for( auto n : current->nodes() ){
            n->setParent(current);
            rewrite(n, i);
            i += i;
        }
    }else
    if(typeid(*root).name() == typeid(BinaryOpNode).name()) {
        
        BinaryOpNode* current = dynamic_cast<BinaryOpNode*>(root);
        current->getChild(0)->setParent(current);
        current->getChild(1)->setParent(current);
        auto leftChild = dynamic_cast<BinaryOpNode*>(current->getChild(0));
        auto rightChild = dynamic_cast<BinaryOpNode*>(current->getChild(1));
        

        rewrite(leftChild, 0);
        rewrite(rightChild, 1);

        if ( current->op() == Add ) {

            // a + b * c
            if ( rightChild != nullptr && rightChild->op() == Multiply ) {
                auto replacementNode = 
                        new MultiplyAddNode(dynamic_cast<ExpressionNode*>(current->getChild(0)), 
                            dynamic_cast<ExpressionNode*>(rightChild->getChild(0)), 
                            dynamic_cast<ExpressionNode*>(rightChild->getChild(1)));
                replaceNode(childIndex, current, replacementNode);
            }
            
            // a * b + c
            if ( leftChild != nullptr && leftChild->op() == Multiply ) {
                auto replacementNode = 
                        new MultiplyAddNode(dynamic_cast<ExpressionNode*>(current->getChild(1)),
                            dynamic_cast<ExpressionNode*>(leftChild->getChild(0)), 
                            dynamic_cast<ExpressionNode*>(leftChild->getChild(1)));
                replaceNode(childIndex, current, replacementNode);
            }

        }
    }else{   
        int numChildren = root->numChildren();
        for ( size_t i = 0; i < numChildren; i++ ) {
            if ( root->getChild(i) != nullptr ){
                root->getChild(i)->setParent(root);
                rewrite(root->getChild(i), i);
            }
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