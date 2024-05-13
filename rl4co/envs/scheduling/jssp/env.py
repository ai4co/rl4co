from rl4co.envs import FJSPEnv

from .generator import JSSPFileGenerator, JSSPwTimeGenerator


class JSSPEnv(FJSPEnv):
    def __init__(
        self,
        generator: JSSPwTimeGenerator = None,
        generator_params: dict = {},
        mask_no_ops: bool = True,
        **kwargs,
    ):
        if generator is None:
            if generator_params.get("file_path", None) is not None:
                generator = JSSPFileGenerator(**generator_params)
            else:
                generator = JSSPwTimeGenerator(**generator_params)

        super().__init__(generator, generator_params, mask_no_ops, **kwargs)
