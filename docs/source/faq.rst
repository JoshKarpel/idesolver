Frequently Asked Questions
==========================

.. currentmodule:: idesolver

How do I install IDESolver?
---------------------------

Installing IDESolver is easy, using `pip <https://pip.pypa.io/en/stable/>`_:

.. code-block:: console

    $ pip install idesolver

I'd like to pickle my IDESolver instance, but it exploded?
----------------------------------------------------------

It probably exploded because you passed ``lambda`` functions as your callables.
You'll need to define the callables somewhere that Python can find them in the global namespace (i.e., top-level functions in a module, or methods in a top-level class, etc.).
