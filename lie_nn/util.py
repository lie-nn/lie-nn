import numpy as np


def vmap(
    fun,
    in_axes=0,
    out_axes=0,
):
    def f(*args):
        in_axes_ = in_axes
        if isinstance(in_axes_, int):
            in_axes_ = (in_axes_,) * len(args)
        assert len(in_axes_) == len(args)

        dims = set()
        for arg, in_axis in zip(args, in_axes_):
            if in_axis is not None:
                dims.add(arg.shape[in_axis])
        assert len(dims) == 1
        dim = dims.pop()

        output = []
        for i in range(dim):
            out = fun(*[arg if in_axis is None else np.take(arg, i, in_axis) for arg, in_axis in zip(args, in_axes_)])
            output.append(out)

        if isinstance(output[0], tuple):
            return tuple(np.stack(list, axis=out_axes) for list in zip(*output))
        return np.stack(output, axis=out_axes)

    return f
