def repeat_batch(x, repeats):
    """Same as repeat on dim=0 for tensordicts as well
    Same as einops.repeat(x, 'b n d -> (r b) n d', r=repeats) but 50% faster
    """
    s = x.shape
    return x.expand(repeats, *s).contiguous().view(s[0] * repeats, *s[1:])


def undo_repeat_batch(x, repeats, dim=0):
    """Undoes repeat_batch
    Same as einops.rearrange(x, '(r b) ... -> r b ...', r=repeats) but 3x faster
    """
    s = x.shape
    return x.view(
        repeats, s[dim] // repeats, *[s[i] for i in range(len(s)) if i != dim]
    )
