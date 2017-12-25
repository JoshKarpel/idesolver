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
date: 12 October 2017
bibliography: paper.bib
---

# Summary

This package provides a general-purpose integro-differential equation solver based on an iterative algorithm devised by Gelmi and Jorquera [@Gelmi2014].
Integro-differential equations appear in many contexts, particularly when trying to describe a system whose current behavior depends on its own history.
A good example is in electronics, where the differential equation for a circuit made only of resistors, capacitors, and inductors can be written in a mixed form where both the derivative and integral of the current appear (for the inductors and capacitors respectively).

Very simple IDEs like those in electronics can often by solved by Laplace transformation (or turned into double-differential form and solved by Fourier transform or any number of other techniques).
Finding analytic solutions for more complicated IDEs is an area of active research and tends to require a special approach for each one.
Even these techniques often produce non-closed forms for the result.

In 2014, Gelmi and Jorquera [@Gelmi2014] published a simple iterative algorithm for finding the solution to a fairly generic integro-differential equation.
IDESolver implements a modified and expanded version of this algorithm.
It can handle both real- and complex-valued IDEs, and uses a slightly different error estimate.


# References

