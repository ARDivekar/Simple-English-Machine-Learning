---
description: 
  An intro to the Chain rule and Multivariable Chain rule of parial derivatives.
---

# Chain rule and Multivariable Chain rule:

## Multivariable Chain rule:

Refs:
- https://www.math.hmc.edu/calculus/tutorials/multichainrule/multichainrule.pdf

### Case with single variable:

Suppose we have functions $$x = f_1(t)$$ and $$y = f_2(t)$$, i.e. each are functions of the variable $$t$$.

Suppose we have another function $$z = f_3(x,y)$$, i.e. $$z$$ is a function of the variables $$x$$ and $$y$$.

We restrict ourselves to the case where $$x$$ and $$y$$ are differentiable at the chosen (but general) point $$(t) \in \mathbb{R}$$, and $$z$$ is differentiable at the corresponding point $$(x,y) \in (\mathbb{R}, \mathbb{R})$$.

By the multivariable chain rule, we have:
$$
\frac{\partial z}{\partial t} = \frac{\partial z}{\partial x} \cdot \frac{\partial x}{\partial t} + \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial t}
$$

The way to remember this rule is: you sum up the multiplication of partial derivatives along each path, and each multiplicative term "cancels out" to the term you require (i.e $$\frac{\partial z}{\partial x} \cdot \frac{\partial x}{\partial t}$$ and $$\frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial t}$$ both "cancel out" to give $$\frac{\partial z}{\partial t}$$, which is what we want to calculate. 

Another way to remember this is: take the **sum of $$\frac{\partial z}{\partial t}$$ along all possible paths from $$z$$ to $$t$$**.

### Case with multiple variables:

Taking a more general case, suppose we have $$x = f_1(a,b)$$ and $$y = f_2(a,b)$$. Once again, $$z = f_3(a_b)$$

Since $$a$$ and $$b$$ are independent of _each other_, this case is exactly the same as the case for a single variable:
$$
\frac{\partial z}{\partial a} = \frac{\partial z}{\partial x} \cdot \frac{\partial x}{\partial a} + \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial a}
$$

and:

$$
\frac{\partial z}{\partial b} = \frac{\partial z}{\partial x} \cdot \frac{\partial x}{\partial b} + \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial b}
$$
