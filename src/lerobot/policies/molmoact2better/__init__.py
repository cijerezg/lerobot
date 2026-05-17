from .configuration_molmoact2better import MolmoAct2BetterConfig

__all__ = [
    "MolmoAct2BetterConfig",
    "MolmoAct2BetterPolicy",
    "make_molmoact2better_pre_post_processors",
]


def __getattr__(name):
    if name == "MolmoAct2BetterPolicy":
        from .modeling_molmoact2better import MolmoAct2BetterPolicy

        return MolmoAct2BetterPolicy
    if name == "make_molmoact2better_pre_post_processors":
        from .processor_molmoact2better import make_molmoact2better_pre_post_processors

        return make_molmoact2better_pre_post_processors
    raise AttributeError(name)
