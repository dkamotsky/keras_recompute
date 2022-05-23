#! -*- coding: utf-8 -*-
# recompute for keras/tf
import tensorflow as tf
from tensorflow.python.util import nest, tf_inspect
from tensorflow.python.eager import tape
from tensorflow.python.ops.custom_gradient import _graph_mode_decorator


def graph_mode_decorator(f, *args, **kwargs):
    return _graph_mode_decorator(f, args, kwargs)


def recompute_grad(call):
    """重计算装饰器（用来装饰Keras层的call函数）
    关于重计算，请参考：https://arxiv.org/abs/1604.06174
    """
    def inner(self, inputs, **kwargs):
        """定义需要求梯度的函数以及重新定义求梯度过程
        （参考自官方自带的tf.recompute_grad函数）
        """
        flat_inputs = nest.flatten(inputs)
        call_args = tf_inspect.getfullargspec(call).args
        for key in ['mask', 'training']:
            if key not in call_args and key in kwargs:
                del kwargs[key]

        def kernel_call():
            """定义前向计算
            """
            return call(self, inputs, **kwargs)

        def call_and_grad(*inputs):
            """定义前向计算和反向计算
            """
            with tape.stop_recording():
                outputs = kernel_call()
                outputs = tf.identity(outputs)

            def grad_fn(doutputs, variables=None):
                watches = list(inputs)
                if variables is not None:
                    watches += list(variables)
                with tf.GradientTape() as t:
                    t.watch(watches)
                    with tf.control_dependencies([doutputs]):
                        outputs = kernel_call()
                grads = t.gradient(outputs, watches, output_gradients=[doutputs])
                del t
                return grads[:len(inputs)], grads[len(inputs):]

            return outputs, grad_fn

        outputs, grad_fn = call_and_grad(*flat_inputs)
        flat_outputs = nest.flatten(outputs)

        def actual_grad_fn(*doutputs):
            grads = grad_fn(*doutputs, variables=self.trainable_weights)
            return grads[0] + grads[1]

        watches = flat_inputs + self.trainable_weights
        watches = [tf.convert_to_tensor(x) for x in watches]
        tape.record_operation(call.__name__, flat_outputs, watches, actual_grad_fn)
        return outputs

    return inner
