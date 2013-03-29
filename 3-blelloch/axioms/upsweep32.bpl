// identity
axiom (forall x : bv32 :: x == A(x,0bv32));
axiom (forall x : bv32 :: x == A(0bv32,x));
// order
axiom (forall x,y : bv32 :: BV32_ULE(x, A(x,y)));
axiom (forall x,y : bv32 :: BV32_ULE(y, A(x,y)));
// associativity
axiom (forall x,y : bv32 :: { A1(x, y) } A1(x, y) == A(x, y));
axiom (forall x,y,z : bv32 :: { A1(x, A(y, z)) } A1(x, A(y, z)) == A(A1(x, y), z));
