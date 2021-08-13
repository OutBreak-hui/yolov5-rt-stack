# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
# Copyright (c) Facebook, Inc. and its affiliates.

from .yolo_runtime import YOLORuntime
from .engine import YOLORTEngine


RUNTIME_REGISTRY = {}
RUNTIME_ENGINE_REGISTRY = {}
ARCH_RUNTIME_REGISTRY = {}
ARCH_RUNTIME_NAME_REGISTRY = {}
ARCH_RUNTIME_INV_REGISTRY = {}
ARCH_CONFIG_REGISTRY = {}


def register_runtime(name, engine=None):
    """
    New runtime types can be added to yolort with the :func:`register_runtime`
    function decorator.

    For example::

        @register_runtime('ncnn')
        class NCNN(YOLORuntime):
            (...)

    Args:
        name (str): the name of the runtime
    """

    def register_runtime_cls(cls):
        if name in RUNTIME_REGISTRY:
            raise ValueError(f"Cannot register duplicate runtime ({name})")
        if not issubclass(cls, YOLORuntime):
            raise ValueError(f"Runtime ({name}: {cls.__name__}) must extend YOLORuntime")
        RUNTIME_REGISTRY[name] = cls
        if engine is not None and not issubclass(engine, YOLORTEngine):
            raise ValueError(f"Engine {engine} must extend YOLORTEngine")

        cls.__engine = engine
        if engine is not None:
            RUNTIME_ENGINE_REGISTRY[name] = engine

            cs = ConfigStore.instance()
            node = engine()
            node._name = name
            cs.store(name=name, group="runtime", node=node, provider="yolort")

            @register_runtime_architecture(name, name)
            def noop(_):
                pass

        return cls

    return register_runtime_cls


def register_runtime_architecture(runtime_name, arch_name):
    """
    New runtime architectures can be added to yolort with the
    :func:`register_runtime_architecture` function decorator. After registration,
    runtime architectures can be selected with the ``--arch`` command-line
    argument.

    For example::

        @register_runtime_architecture('ncnn', 'ncnn_x86_64')
        def ncnn_x86_64(cfg):
            args.transform = getattr(cfg.runtime, 'transform', Transform(...))
            (...)

    The decorated function should take a single argument *cfg*, which is a
    :class:`omegaconf.DictConfig`. The decorated function should modify these
    arguments in-place to match the desired architecture.

    Args:
        runtime_name (str): the name of the Runtime (Runtime must already be registered)
        arch_name (str): the name of the runtime architecture (``--arch``)
    """

    def register_runtime_arch_fn(fn):
        if runtime_name not in RUNTIME_REGISTRY:
            raise ValueError("Cannot register runtime architecture for "
                             f"unknown runtime type ({runtime_name})")
        if arch_name in ARCH_RUNTIME_REGISTRY:
            raise ValueError(f"Cannot register duplicate runtime architecture ({arch_name})")
        if not callable(fn):
            raise ValueError(f"Runtime architecture must be callable ({arch_name})")
        ARCH_RUNTIME_REGISTRY[arch_name] = RUNTIME_REGISTRY[runtime_name]
        ARCH_RUNTIME_NAME_REGISTRY[arch_name] = runtime_name
        ARCH_RUNTIME_INV_REGISTRY.setdefault(runtime_name, []).append(arch_name)
        ARCH_CONFIG_REGISTRY[arch_name] = fn
        return fn

    return register_runtime_arch_fn
