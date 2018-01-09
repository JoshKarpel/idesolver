---
title: 'idesolver: a general purpose integro-differential equation (IDE) solver'
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
date: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
bibliography: paper.bib
---

# Summary

This package provides a general-purpose integro-differential equation solver based on an iterative algorithm devised by Gelmi and Jorquera [@Gelmi2014].
Integro-differential equations appear in many contexts, particularly when trying to describe a system whose current behavior depends on its own history.
A common example is in electronics, where the differential equation for a circuit made only of resistors, capacitors, and inductors can be written in a mixed form where both the derivative and integral of the current appear (for the inductors and capacitors respectively).

Very simple IDEs like those in electronics can often by solved by Laplace transformation (or turned into double-differential form and solved by Fourier transform or other techniques).
Finding analytic solutions for more complicated IDEs is an area of active research and tends to require a special approach for each one.
Even these techniques often produce non-closed forms for the result, which can be difficult to apply practically depending on their convergence properties.

In 2014, Gelmi and Jorquera [@Gelmi2014] published a simple iterative algorithm for finding the solution to a fairly generic integro-differential equation.
IDESolver implements a modified and expanded version of this algorithm.
It can handle both real-valued and complex-valued IDEs and allows for configuration of the error estimator.

The algorithm produces a new guess of the solution by solving the IDE as if it was an ODE (ordinary differential equation) by pretending that the derivative of the new guess is given by the integral equation, but with the old guess substituted for the new guess.
This means that there is never a need to solve the actual IDE, just a series of ODEs.
Once the guess stops changing dramatically (according to some tolerance set by the user) or a maximum number of iterations is reached (again, set by the user) the algorithm terminates.

# References

