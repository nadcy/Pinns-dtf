import ast
c = 5
eq_ast = ast.parse('diff(y,t)+dydx')
eq_exp = eq_ast.body[0]
print(str(eq_exp))
ast.Expr
li = ast.walk(ast.Expr)

while True:
    try:
        print (next(li))
    except StopIteration:
        break

print(isinstance(eq_exp,ast.AST))
for i in li:
    print(i)
print(ast.dump(eq_exp, indent=4))
print(ast.dump(eq_ast, indent=4))