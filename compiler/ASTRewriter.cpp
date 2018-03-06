#include "ASTRewriter.hpp"
//#include <iostream>
//#include <stdexcept>
//#include <sstream>
#include <typeinfo>


// Helper function for replacing the current node.
// The function assumes that the replacementNode already has received currentNode's children.
void replaceNode(const int childIndex, Node* currentNode, Node* replacementNode)
{

    // The node's children needs to be nullptrs before deletion.
    // If they aren't the children will be deleted too.
    for ( int i = 0; i < currentNode->numChildren(); i++ ){
        currentNode->setChild(i,nullptr);
    }

    //if( currentNode->getParent() != nullptr ){
        currentNode->getParent()->setChild(childIndex,replacementNode);
    //}
    
    delete currentNode;
}


// The rewriter traverses and modifies the subtree of the root node in a pre-order fashion.
// This means that the algorithm first recurses to the bottom of the tree and modifies the
// tree from the bottom and up.
// The childIndex parameter denotes the placement of root relative to its parent.
// childIndex = 0 means it's the first child, childIndex means it's the second child.
void ASTRewriter::rewrite(Node* root, const int childIndex)
{    

    if(root == nullptr) { return; }

    if(typeid(*root).name() == typeid(SequenceNode).name()) {
        auto current = dynamic_cast<SequenceNode*>(root);
        int i = 0;
        for( auto n : current->nodes() ){
            n->setParent(current);
            rewrite(n, i);
            ++i;
        }
    }else
    // Pattern match for binary operations
    if(typeid(*root).name() == typeid(BinaryOpNode).name()) {
        
        BinaryOpNode* current = dynamic_cast<BinaryOpNode*>(root);
    
        current->getChild(0)->setParent(current);
        current->getChild(1)->setParent(current);
        rewrite(current->getChild(0), 0);
        rewrite(current->getChild(1), 1);

        // Pattern matching for multiply-add
        if ( current->op() == Add ) {
            
            auto lhs = dynamic_cast<BinaryOpNode*>(current->getChild(0));
            auto rhs = dynamic_cast<BinaryOpNode*>(current->getChild(1));


            // Accounts for a*b+c and a+b*c
            BinaryOpNode* mulOpNode;
            int index;
            if (lhs != nullptr && lhs->op() == Multiply) {
                mulOpNode = lhs;
                // child index of rhs
                index = 1;
            }else
            if (rhs != nullptr && rhs->op() == Multiply) {
                mulOpNode = rhs;
                //child index of lhs
                index = 0;
            } else {
                return;
            }

            auto replacementNode = 
                        new MultiplyAddNode(dynamic_cast<ExpressionNode*>(mulOpNode->getChild(0)),
                                            dynamic_cast<ExpressionNode*>(mulOpNode->getChild(1)),
                                            dynamic_cast<ExpressionNode*>(current->getChild(index)));
            replaceNode(childIndex, current, replacementNode);
        }
    } else {   
        for( int i = 0; i < root->numChildren(); i++ ) {
            if( root->getChild(i) != nullptr ){
                root->getChild(i)->setParent(root);
                rewrite(root->getChild(i), i);
            }
        }
    }
}