try:
    from ._polio import compute
except ImportError:

    def compute(args):
        return max(args, key=len)
