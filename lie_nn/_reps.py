import jax

from .groups import AbstractRep as Rep
from .groups import static_jax_pytree


@static_jax_pytree
class MulRep:
    mul: int
    rep: Rep

    @property
    def dim(self):
        return self.mul * self.rep.dim

    def __iter__(self):
        yield self.mul
        yield self.rep

    def __repr__(self):
        return f"{self.mul}x{self.rep}"


class Reps(tuple):
    def __new__(cls, reps=None):
        if isinstance(reps, Reps):
            return super().__new__(cls, reps)

        list = []
        if isinstance(reps, Rep):
            list.append(MulRep(1, reps))
        elif reps is None:
            pass
        else:
            for mul_rep in reps:
                if isinstance(mul_rep, Rep):
                    mul = 1
                    rep = mul_rep
                elif isinstance(mul_rep, MulRep):
                    mul, rep = mul_rep
                elif len(mul_rep) == 2:
                    mul, rep = mul_rep
                    assert isinstance(rep, Rep)
                else:
                    mul = None
                    rep = None

                if not (isinstance(mul, int) and mul >= 0 and rep is not None):
                    raise ValueError(f'Unable to interpret "{mul_rep}" as a MulRep.')

                list.append(MulRep(mul=mul, rep=rep))
        return super().__new__(cls, list)

    def slices(self):
        r"""List of slices corresponding to indices for each irrep."""
        s = []
        i = 0
        for mul_rep in self:
            s.append(slice(i, i + mul_rep.dim))
            i += mul_rep.dim
        return s

    def __getitem__(self, i):
        x = super().__getitem__(i)
        if isinstance(i, slice):
            return Reps(x)
        return x

    def __contains__(self, rep) -> bool:
        assert isinstance(rep, Rep)
        return rep in (rep for _, rep in self)

    def count(self, rep) -> int:
        r"""Multiplicity of ``rep``."""
        assert isinstance(rep, Rep)
        return sum(mul for mul, rep2 in self if rep == rep2)

    def index(self, _):
        raise NotImplementedError

    def __add__(self, reps):
        reps = Reps(reps)
        return Reps(super().__add__(reps))

    @property
    def dim(self) -> int:
        return sum(mul_rep.dim for mul_rep in self)

    def __repr__(self):
        return "+".join(f"{mul_rep}" for mul_rep in self)


jax.tree_util.register_pytree_node(Reps, lambda x: ((), x), lambda x, _: x)
