function {:bvbuiltin "bvule"} BV16_ULE(bv16, bv16) : bool;

// identity
axiom (forall x : bv16 :: x == A(x,0bv16));
axiom (forall x : bv16 :: x == A(0bv16,x));
// order
axiom (forall x,y : bv16 :: BV16_ULE(x, A(x,y)));
axiom (forall x,y : bv16 :: BV16_ULE(y, A(x,y)));
// associativity
axiom (forall x,y : bv16 :: { A1(x, y) } A1(x, y) == A(x, y));
axiom (forall x,y,z : bv16 :: { A1(x, A(y, z)) } A1(x, A(y, z)) == A(A1(x, y), z));
