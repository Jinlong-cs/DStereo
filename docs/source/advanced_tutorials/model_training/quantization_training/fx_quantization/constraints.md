# symbolic trace 支持的操作说明

symbolic trace 的一大特征是在 trace 时并不实际执行代码，而是仅对模型中的操作进行记录。实际上，在 symbolic trace 的过程中，fx 使用 `Proxy` 作为数据流的载体，记录对数据流的操作，并生成相应的结点。也就是说，fx 使用 Proxy 作为**模型输入**，并作为**叶子结点的输出**，这也就要求模型输入和叶子结点的输出必须使用 `Proxy` 可以模拟的数据结构，`Proxy` 的定义如下，symbolic trace 仅能处理在 `Proxy` 中显式支持的操作，其他操作默认不支持

*在使用 wrap 时也要注意，被包装的 callable 的输入输出必须是 Proxy 可以模拟的数据结构*

```python
class Proxy:
    """
    ``Proxy`` objects are ``Node`` wrappers that flow through the
    program during symbolic tracing and record all the operations
    (``torch`` function calls, method calls, operators) that they touch
    into the growing FX Graph.

    For a more detailed description into the Proxy internals, check out
    the "Proxy" section in `torch/fx/OVERVIEW.md`
    """

    @compatibility(is_backward_compatible=True)
    def __init__(self, node: Node, tracer: 'Optional[TracerBase]' = None):
        if tracer is None:
            # This allows you to create a Proxy object around a raw Node
            tracer = GraphAppendingTracer(node.graph)
        self.tracer = tracer
        self.node = node

    def __repr__(self) -> str:
        return f'Proxy({self.node.name})'

    # 支持 getattr， Attribute 在 __call__ 会生成 call_method 结点，从而
    # 支持 x.abs() 这种操作
    def __getattr__(self, k) -> 'Attribute':
        # note: not added to the graph yet, if this is a method call
        # we peephole optimize to the method invocation
        return Attribute(self, k)

    # 支持 __call__
    def __call__(self, *args, **kwargs) -> 'Proxy':
        return self.tracer.create_proxy('call_method', '__call__', (self,) + args, kwargs)

    # self.tracer.iter 默认会抛出 TraceError，因为无法确定迭代长度
    def __iter__(self) -> Iterable['Proxy']:
        # 此处 UNPACK_SEQUENCE 相关逻辑暂未找到说明，可能在开发中或仅供内部使用
        frame = inspect.currentframe()
        assert frame is not None
        calling_frame = frame.f_back
        assert calling_frame is not None
        inst = list(dis.get_instructions(calling_frame.f_code))[calling_frame.f_lasti // 2]
        if inst.opname == 'UNPACK_SEQUENCE':
            return (self[i] for i in range(inst.argval))  # type: ignore[index]

        return self.tracer.iter(self)

    # self.tracer.to_bool 默认会抛出 TraceError，因为无法确定真值
    def __bool__(self) -> bool:
        return self.tracer.to_bool(self)

    # self.tracer.keys 内部会 return Attribute(proxy, 'keys')()，
    # 从而支持 x.keys()
    @compatibility(is_backward_compatible=True)
    def keys(self):
        return self.tracer.keys(self)

    # 默认不支持 len
    def __len__(self):
        raise RuntimeError("'len' is not supported in symbolic tracing by default. If you want "
                           "this call to be recorded, please call torch.fx.wrap('len') at "
                           "module scope")

    # 通过此方式支持 torch 的所有 function 形式的算子
    # torch.xxx & torch.nn.functional.xxx
    # 相关机制见官方文档
    # https://pytorch.org/docs/1.10/notes/extending.html#extending-torch
    @classmethod
    def __torch_function__(cls, orig_method, types, args=None, kwargs=None):
        args = args if args else ()
        kwargs = kwargs if kwargs else {}

        tracers : Dict[Any, None] = {}

        def find_tracer(a):
            if isinstance(a, cls):
                tracers[a.tracer] = None
        torch.fx.node.map_aggregate(args, find_tracer)
        torch.fx.node.map_aggregate(kwargs, find_tracer)

        if len(tracers) > 1:
            raise RuntimeError(f'Found multiple different tracers {list(tracers.keys())} while '
                               f'trying to trace operations {orig_method}')
        tracer = next(iter(tracers.keys()))

        if isinstance(orig_method, torch._C.ScriptMethod):
            args = (orig_method.owner,) + args
            return tracer.create_proxy('call_method', orig_method.name, args, kwargs)
        if torch.overrides.is_tensor_method_or_property(orig_method):
            return tracer.create_proxy('call_method', orig_method.__name__, args, kwargs)
        else:
            return tracer.create_proxy('call_function', orig_method, args, kwargs,
                                       name=tracer.graph._target_to_str(orig_method.__name__))

reflectable_magic_methods = {
    'add': '{} + {}',
    'sub': '{} - {}',
    'mul': '{} * {}',
    'floordiv': '{} // {}',
    'truediv': '{} / {}',
    'div': '{} / {}',
    'mod': '{} % {}',
    'pow': '{} ** {}',
    'lshift': '{} << {}',
    'rshift': '{} >> {}',
    'and_': '{} & {}',
    'or_': '{} | {}',
    'xor': '{} ^ {}',
    'getitem': '{}[{}]',
    'matmul': '{} @ {}',
}

magic_methods = dict({
    'eq': '{} == {}',
    'ne': '{} != {}',
    'lt': '{} < {}',
    'gt': '{} > {}',
    'le': '{} <= {}',
    'ge': '{} >= {}',
    'pos': '+{}',
    'neg': '-{}',
    'invert': '~{}'}, **reflectable_magic_methods)

# 通过注册 magic method 的方式支持 magic_methods 中的 operators
for method in magic_methods:
    def _scope(method):
        def impl(*args, **kwargs):
            tracer = args[0].tracer
            target = getattr(operator, method)
            return tracer.create_proxy('call_function', target, args, kwargs)
        impl.__name__ = method
        as_magic = f'__{method.strip("_")}__'
        setattr(Proxy, as_magic, impl)
    _scope(method)

# 支持将 Proxy 作为右值
def _define_reflectable(orig_method_name):
    method_name = f'__r{orig_method_name.strip("_")}__'

    def impl(self, rhs):
        target = getattr(operator, orig_method_name)
        return self.tracer.create_proxy('call_function', target, (rhs, self), {})
    impl.__name__ = method_name
    impl.__qualname__ = method_name
    setattr(Proxy, method_name, impl)

for orig_method_name in reflectable_magic_methods:
    _define_reflectable(orig_method_name)
```

## nn.Module

 `torch.nn` 下除 `ModuleList`、`ModuleDict`、`Sequantial` 以外的其他算子，在 symbolic trace 中默认会被视为叶子结点，生成 call_module 类型的 Node

## 控制流

受限于符号执行的机制，symbolic trace 不支持动态控制流（控制条件依赖模型输入，trace 时将直接报错），且静态控制流在 trace 之后将固定下来无法再修改（条件语句将固定执行 trace 时执行到的分支，循环语句的循环次数将被固定）

***注意，模型中对于 `self.training` 状态的判断属于静态控制流***
