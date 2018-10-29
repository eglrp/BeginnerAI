import numpy as np
import functools as ft
import math

if 'GLOBAL_VARIABLE_SCOPE' not in globals():
    GLOBAL_VARIABLE_SCOPE = {}

class Jensor(object):
    def __init__(self, name):
        self.name = name

        self.prev = []
        self.next = []

class Variable(Jensor):
    initial = 'MSRA'
    method = 'SGD'
    def __init__(self, shape=list, name=str, scope='', grad=True, learnable=False):
        if scope != '':
            self.scope = scope if scope[-1] == '/' else scope + '/'
            self.name = self.scope + name
        else:
            self.name = name
            self.scope = scope
        super(Variable, self).__init__(self.name)

        if self.name in GLOBAL_VARIABLE_SCOPE:
            raise Exception('Variable name: %s exists!' % self.name)
        else:
            GLOBAL_VARIABLE_SCOPE[self.name] = self

        for i in shape:
            if not isinstance(i, int):
                raise Exception("Variable name: %s shape is not list of int"%self.name)

        self.shape = shape
        self.data = self._initializer(shape, self.initial)

        if grad:
            self.diff = np.zeros(self.shape)
            self.wait_bp = True
            self.learnable = learnable

    def _initializer(self, shape, method):
        if method == 'const':
            return np.random.standard_normal(shape) / 100

        if method == 'None':
            return np.zeros(shape)

        if method == 'MSRA':
            weights_scale = math.sqrt(ft.reduce(lambda x, y: x * y, shape) / shape[-1])
            return np.random.standard_normal(shape) / weights_scale

    def eval(self):
        for operator in self.prev:
            GLOBAL_VARIABLE_SCOPE[operator].forward()
        self.wait_bp = True
        return self.data

    def diff_eval(self):
        if self.wait_bp:
            for operator in self.next:
                GLOBAL_VARIABLE_SCOPE[operator].backward()
            self.wait_bp = False
        else:
            pass

        return self.diff

    def apply_gradient(self, learning_rate=float, decay_rate=float, batch_size=1):
        self.data *= (1 - decay_rate)
        if self.method == 'SGD':
            learning_rate = learning_rate
            self.data -= (learning_rate*self.diff/batch_size)
            self.diff *= 0

        elif self.method == 'Momentum':
            self.mtmp = self.momentum * self.mtmp + self.diff/batch_size
            self.data -= learning_rate * self.mtmp
            self.diff *= 0

        elif self.method == 'NGA':
            self.mtmp = self.momentum * self.mtmp + self.diff / batch_size + self.momentum*(self.diff-self.lastdiff)/batch_size
            self.data -= learning_rate * self.mtmp
            self.lastdiff = self.diff
            self.diff *= 0

        elif self.method == 'Adam':
            self.t += 1
            learning_rate_t = learning_rate * math.sqrt(1 - pow(self.beta2, self.t)) / (1 - pow(self.beta1, self.t))
            self.m_t = self.beta1 * self.m_t + (1 - self.beta1) * self.diff / batch_size
            self.v_t = self.beta2 * self.v_t + (1 - self.beta2) * ((self.diff / batch_size) ** 2)
            self.data -= learning_rate_t * self.m_t / (self.v_t + self.epsilon) ** 0.5
            self.diff *= 0

        else:
            raise Exception('No apply_gradient method: %s'%self.method)

    def set_method_sgd(self):
        self.method = 'SGD'

    def set_method_momentum(self, momentum=0.9):
        self.method = 'Momentum'
        self.momentum = momentum
        self.mtmp = np.zeros(self.diff.shape)

    def set_method_nga(self,momentum=0.9):
        self.method = 'NGA'
        self.lastdiff = np.zeros(self.diff.shape)
        self.momentum= momentum
        self.mtmp = np.zeros(self.diff.shape)

    def set_method_adam(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.method = 'Adam'
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_t = np.zeros(self.diff.shape)
        self.v_t = np.zeros(self.diff.shape)
        self.t = 0

class Operator(Jensor):
    def __init__(self, name, input_variables, output_variables):
        super(Operator, self).__init__(name)

        if name in GLOBAL_VARIABLE_SCOPE.keys():
            raise Exception("Operator %s has exists !"%name)

        if not isinstance(input_variables, Variable) and not isinstance(input_variables[0], Variable):
            raise Exception("Operator %s 's input_variables is not instance(or list) of Variable!")

        if not isinstance(output_variables, Variable) and not isinstance(output_variables[0], Variable):
            raise Exception("Operator %s 's output_variables is not instance(or list) of Variable!")

        GLOBAL_VARIABLE_SCOPE[self.name] = self

        # register for input Variable's child and output Variable's parents
        self._register_graph(input_variables, output_variables, self)

        self.wait_forward = True
        # self.wait_backward = not self.wait_forward

    def forward(self):
        pass

    def backward(self):
        pass

    def _register_graph(self, input_variable, output_variable, operator):
        if isinstance(input_variable,Variable) and isinstance(output_variable, Variable):
            input_variable.next.append(operator.name)
            output_variable.prev.append(operator.name)
            operator.prev.append(input_variable.name)
            operator.next.append(output_variable.name)

        elif isinstance(input_variable, Variable) and len(output_variable)>1:
            for output in output_variable:
                input_variable.next.append(operator.name)
                output.prev.append(operator.name)
                operator.prev.append(input_variable.name)
                operator.next.append(output.name)

        elif isinstance(output_variable, Variable) and len(input_variable)>1:
            for _input in input_variable:
                _input.next.append(operator.name)
                output_variable.prev.append(operator.name)
                operator.prev.append(_input.name)
                operator.next.append(output_variable.name)

        elif len(output_variable)> 1 and len(input_variable)> 1:
            for _input in input_variable:
                _input.next.append(operator.name)
                operator.prev.append(_input.name)
            for output in output_variable:
                output.prev.append(operator.name)
                operator.next.append(output.name)

        else:
            raise Exception('Operator name %s input,output list error'% operator.name)
