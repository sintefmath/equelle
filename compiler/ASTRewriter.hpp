#ifndef ASTREWRITER_HEADER_INCLUDED
#define ASTREWRITER_HEADER_INCLUDED

class SequenceNode;
class NumberNode;
class QuantityNode;
class BasicUnitNode;
class BinaryOpUnitNode;
class PowerUnitNode;
class StringNode;
class TypeNode;
class CollectionTypeNode;
class FuncTypeNode;
class BinaryOpNode;
class ComparisonOpNode;
class NormNode;
class UnaryNegationNode;
class OnNode;
class TrinaryIfNode;
class VarDeclNode;
class VarAssignNode;
class VarNode;
class FuncRefNode;
class JustAnIdentifierNode;
class FuncArgsDeclNode;
class FuncDeclNode;
class FuncStartNode;
class FuncAssignNode;
class FuncArgsNode;
class ReturnStatementNode;
class FuncCallNode;
class FuncCallStatementNode;
class LoopNode;
class ArrayNode;
class RandomAccessNode;
class StencilAssignmentNode;
class StencilNode;
class MultiplyAddNode;


class ASTRewriter
{
public:
	ASTRewriter()
	{
	};

    ~ASTRewriter()
    {
    };

    void rewrite(Node* root, const childIndex);
};

#endif // ASTREWRITER_HEADER_INCLUDED
