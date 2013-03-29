function {:bvbuiltin "bvule"} BV8_ULE(bv8, bv8) : bool;

// identity
axiom (forall x : bv8 :: x == A(x,0bv8));
axiom (forall x : bv8 :: x == A(0bv8,x));
// order
axiom (forall x,y : bv8 :: BV8_ULE(x, A(x,y)));
axiom (forall x,y : bv8 :: BV8_ULE(y, A(x,y)));
// associativity
axiom (forall x,y : bv8 :: { A1(x, y) } A1(x, y) == A(x, y));
axiom (forall x,y,z : bv8 :: { A1(x, A(y, z)) } A1(x, A(y, z)) == A(A1(x, y), z));
