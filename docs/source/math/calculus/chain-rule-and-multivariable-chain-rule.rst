===================================================================================================
Chain rule and Multivariable Chain rule
===================================================================================================

.. _multivariable-chain-rule:

Multivariable Chain rule
===================================================================================================

Refs
---------------------------------------------------------------------------------------------------

- :download:`https://www.usna.edu/Users/oceano/raylee/SM223/Ch14_5_Stewart(2016).pdf </_static/resources/other/Multivariable chain rule Ch14_5_Stewart(2016).pdf>`


.. _multivariable-chain-rule-with-a-single-variable:

Multivariable Chain rule (with a single input variable)
---------------------------------------------------------------------------------------------------

Suppose we have functions :math:`x = f_1(t)` and :math:`y = f_2(t)`, i.e. each are functions of the variable :math:`t`.

Suppose we have another function :math:`z = f_3(x,y)`, i.e. :math:`z` is a function of the variables :math:`x` and :math:`y`.

We restrict ourselves to the case where :math:`x` and :math:`y` are differentiable at the chosen (but general) point :math:`t \in \mathbb{R}`, and :math:`z` is differentiable at the corresponding point :math:`(x, y) \in (\mathbb{R}, \mathbb{R})`.

By the multivariable chain rule, we have:

.. math::

    \frac{\partial z}{\partial t} = \frac{\partial z}{\partial x} \cdot \frac{\partial x}{\partial t} + \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial t}

.. figure:: /_static/img/calculus/Multivariable-chain-rule.svg
    :alt: Multivariable chain rule
    :width: 400pt

    Multivariable chain rule

One way to remember this rule: 

    Starting at the final variable (:math:`z`), you go along each path to the input variable (:math:`t`), and multiply every partial derivative along the path. Each multiplicative term "cancels out" to the term you require (i.e :math:`\frac{\partial z}{\partial x} \cdot \frac{\partial x}{\partial t}` "cancels out" to give :math:`\frac{\partial z}{\partial t}`, which is what we want to calculate. :math:`\frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial t}` does the same). Finally, you add together all the chains of multiplications, which gives us the result above.

In short: take the **sum of multiplications which simplify to** :math:`\frac{\partial z}{\partial t}`, **along all possible paths from** :math:`z` **to** :math:`t`.


.. _multivariable-chain-rule-with-a-multiple-variables:

Multivariable Chain rule (with multiple unrelated input variables)
---------------------------------------------------------------------------------------------------


Taking a more general case, suppose we have :math:`x = f_1(a,b)` and :math:`y = f_2(a,b)`. Once again, :math:`z = f_3(x, y)`

Since the base variables :math:`a` and :math:`b` have no dependencies between *each other*, this case is exactly the same as the case for a single variable:

.. math::

    \frac{\partial z}{\partial a} = \frac{\partial z}{\partial x} \cdot \frac{\partial x}{\partial a} + \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial a}


and:

.. math::

    \frac{\partial z}{\partial b} = \frac{\partial z}{\partial x} \cdot \frac{\partial x}{\partial b} + \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial b}
