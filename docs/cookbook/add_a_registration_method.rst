.. _cookbook_new_registrar:

============================
Adding a Registration Method
============================

Wire a new deformable image-registration backend into MONAI Physio so every
workflow and CLI that takes a ``registration_method`` can use it.

Ingredients
===========

* **An algorithm** that aligns a moving ITK image to a fixed ITK image and can
  express the result as ITK transforms.
* **Both directions.** The backend must produce the forward *and* inverse
  transform. If it natively gives only one, invert it before returning.
* **A module** at ``src/monai_physio/register_images_<name>.py``.

Steps
=====

**1. Subclass** :class:`~monai_physio.RegisterImagesBase`. The base owns
preprocessing, mask handling and dilation, modality settings, transform
composition, and the public ``register()`` / ``register_from()`` API. You
supply only the solve.

.. code-block:: python

   import logging
   from typing import Optional, Union

   import itk

   from monai_physio import RegisterImagesBase


   class RegisterImagesMyMethod(RegisterImagesBase):
       def __init__(self, log_level: int | str = logging.INFO) -> None:
           super().__init__(log_level=log_level)
           self.number_of_iterations = 50

       def registration_method(
           self,
           moving_image: itk.Image,
           moving_mask: Optional[itk.Image] = None,
           moving_labelmap: Optional[itk.Image] = None,
           moving_image_pre: Optional[itk.Image] = None,
       ) -> dict[str, Union[itk.Transform, float]]:
           if self.fixed_image_pre is None:
               raise ValueError("Fixed image must be set before registration.")

           moving_pre = moving_image_pre if moving_image_pre is not None else moving_image
           forward, inverse, loss = solve(self.fixed_image_pre, moving_pre)

           return {
               "fixed_to_moving_transform": forward,
               "moving_to_fixed_transform": inverse,
               "loss": loss,
           }

``registration_method()`` is the one required override. It is internal -
callers use ``register()``, which wraps it.

**2. Honor the contract.** ``fixed_to_moving_transform`` warps the *moving
image* onto the fixed grid. ``moving_to_fixed_transform`` warps *moving
points* into fixed space. Images and points take opposite transforms; getting
this backwards is the classic silent failure here. Read
:doc:`/developer/transform_conventions` before you return anything.

**3. Read the base's state, don't re-derive it.** Use
``self.fixed_image_pre`` and ``moving_image_pre`` rather than preprocessing
again, ``self.fixed_mask`` / ``self.moving_mask`` for ROI-limited solves, and
``self.modality`` for modality-specific parameters. Honor ``self.fast_mode``
by dropping to cheaper settings - automated tests rely on it.

**4. Do not accept an initial transform.** Seeding is the base class's job:
``register_from()`` pre-warps the moving image, calls your method on the
residual, and composes. One implementation, identical for every backend.

**5. Export it** from ``src/monai_physio/__init__.py``, next to the other
``register_images_*`` imports, and add it to ``__all__``.

**6. Add it to the CLI dispatch** in
``src/monai_physio/cli/_method_factories.py`` - the single place strings become
instances. Append the name to ``REGISTRATION_METHODS`` and a branch to
``build_registration_method()``. Every ``--registration-method`` flag picks it
up from there.

**7. Test it.** Copy the shape of ``tests/test_register_images_greedy.py``.
Register a synthetic image against a known warp of itself and assert the
recovered transform, in both directions. Mark GPU-bound tests ``requires_gpu``.

.. code-block:: bash

   py -m pytest tests/test_register_images_my_method.py -v

**8. Document it.** Add ``docs/api/registration/<name>.rst``, list it in that
directory's ``index.rst``, and mention it in
:doc:`/developer/registration_images`. Then run ``graphify update .``.

Notes
=====

* Your backend composes for free:
  :class:`~monai_physio.RegisterImagesChain` will run it as one stage of a
  multi-stage pipeline, and
  :class:`~monai_physio.RegisterTimeSeriesImages` will apply it across a whole
  4D series, without either class knowing about it.
* Use ``TransformTools.transform_image()`` and
  ``TransformTools.transform_pvcontour()`` to apply results - they encode the
  direction rules.
* Registering *models* to patients is a different base class; see
  :doc:`/developer/registration_models`.

See Also
========

* :doc:`/developer/registration_images` - the extended guide
* :doc:`/developer/transform_conventions` - required reading
* :doc:`/api/registration/base`
* :doc:`add_a_segmentation_method`
