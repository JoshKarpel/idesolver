---
title: 'IDESolver: a general purpose integro-differential equation solver'
tags:
  - Python
  - Integro-Differential Equations
authors:
 - name: Joshua T Karpel
   orcid: 0000-0001-5968-9373
   affiliation: 1
affiliations:
 - name: University of Wisconsin - Madison
   index: 1
date: 15 January 2018
bibliography: paper.bib
---

# Summary

IDESolver provides a general-purpose numerical integro-differential equation (IDE) solver based on an iterative algorithm devised by Gelmi and Jorquera [@Gelmi2014].
IDEs appear in many contexts, particularly when trying to describe a system whose current behavior depends on its own history.
A common example is in electronics, where the governing equation for a circuit made only of resistors, capacitors, and inductors can be written in a mixed form where both the derivative and integral of the current appear (for the inductors and capacitors respectively).
More complicated examples may contain convolutions of the unknown function against some kernel function, or even be nonlinear in the unknown function.

Simple IDEs can often by solved using integral transforms.
For example, Laplace and Fourier transforms are often used to solve simple circuit problems analytically.
Finding analytic solutions for more complicated IDEs is an area of active research and tends to require a special approach for each one.
Even these techniques often produce non-closed forms for the result, which can be difficult to apply practically depending on their convergence properties.
The details of these methods are often far more complicated than a researcher who just wants a numerical solution to their IDE will want to deal with.

In 2014, Gelmi and Jorquera [@Gelmi2014] published a simple and robust algorithm for finding numeric solutions of generic integro-differential equations.
IDESolver implements a modified and expanded version of this algorithm as a Python library.
It handles both real-valued and complex-valued IDEs, allows for configuration of the error estimator, and provides control over the error tolerances of the internal parts of the algorithm through a convenient interface.
The typical user should not need to think about the methodology of solving their IDE at all (but advanced users can provide alternate subroutines if desired).

# The Algorithm

Gelmi and Jorquera's algorithm is conceptually similar to iterative approximation methods in linear algebra [@Axelsson1996], but repurposed for use with an IDE.
At each step it produces a new guess of the solution by solving the IDE as if it was an ODE (ordinary differential equation), but using the previous guess to evaluate the new derivative.
This means that there is never a need to solve the actual IDE, just a series of ODEs.
Once the guess stops changing dramatically (according to some tolerance set by the user) or a maximum number of iterations is reached (again, set by the user) the algorithm terminates.

# References

