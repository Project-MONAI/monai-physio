"""Parameters for the physics-informed Duke heart tutorials (16, 17 and 18).

These tutorials train the same cardiac motion surrogate Tutorials 9 and 10 do,
but price its predictions against a neo-Hookean strain energy as well as against
measured displacement.  That energy needs volume elements, so they build their
own *tetrahedral* shape model rather than reusing the surface one Tutorials 6 to
8 write.  Everything they produce therefore lives in its own directories, and
Tutorials 1 to 15 are neither read from nor written to.

The cohort itself is unchanged, so this derives from
:class:`parameters_duke_heart_labelmaps.ParametersDukeHeartLabelmaps` and adds
only what is new: the element size of the volumetric model and the constitutive
law.  The held-out case, the label ids, the ICP transform type and the greedy
schedule are inherited, which is what keeps the two chains comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from parameters_duke_heart_labelmaps import ParametersDukeHeartLabelmaps


@dataclass(frozen=True)
class ParametersDukeHeartPhysicsInformed(ParametersDukeHeartLabelmaps):
    """Settings the physics-informed Duke heart tutorials add to the cohort's.

    Attributes:
        ssm_element_size_mm: Edge length of the tetrahedra filling the shape
            model's template.  Distinct from the inherited
            ``mesh_element_size_mm`` -- which happens to carry the same value
            but sizes the per-case meshes Tutorial 4 writes for quality
            assurance, and which nothing downstream reads -- because this one
            sizes the elements the strain energy is integrated over.

            It is a coverage/cost trade, and the numbers are measured rather
            than assumed: ``extract_tetrahedra`` resamples the mask with a vote,
            so any wall thinner than the element size is dropped, and the thin
            atrial and right-ventricular walls go first.  Against the 208 259
            mm^3 the mean surface encloses, the template holds 99.5% of it at
            1.0 mm (305,696 nodes), 88.3% at 1.5 mm (100,903 nodes), 72.1% at
            2.0 mm (43,826 nodes) and 61.9% at 2.5 mm.  1.5 mm keeps the left
            ventricle and most of the right, at a graph roughly five times the
            20,000-point surface one Tutorial 9 trains on -- which is the regime
            ``TrainPhysicsNeMoMGN.set_num_processor_checkpoint_segments`` exists
            for, so expect to enable it and to lower ``batch_size``.
        mu_kpa: Neo-Hookean shear modulus, in kilopascals.  Passive myocardium
            in diastole, which is the state the reference frame is taken in.
        lambda_lame_kpa: First Lame parameter, in kilopascals.  Ten times
            ``mu_kpa``, making the tissue stiff against volume change without
            imposing the exact incompressibility that would lock elements this
            size.
        lambda_physics: Weight of the physics residual against the displacement
            loss.  The two are not in the same units -- displacement is scored
            normalized, the residual in kilopascals -- so this is a value to
            sweep rather than a value to trust.
        number_of_epochs: Training epochs, matching Tutorial 9 so the two are
            comparable.
        number_of_epochs_test: Same, under ``TestTools.running_as_test``.
        train_ablation_baseline: Whether Tutorial 17 also trains a second model
            with ``lambda_physics`` at zero.  That model sees exactly the same
            volumetric data, so it is the only comparison that isolates the
            physics term; measuring against Tutorial 9 instead would confound it
            with the change from a surface shape model to a volumetric one.  It
            costs a second training run.
    """

    ssm_element_size_mm: float = 1.5

    mu_kpa: float = 10.0
    lambda_lame_kpa: float = 100.0
    lambda_physics: float = 0.1

    number_of_epochs: int = 1500
    number_of_epochs_test: int = 2

    train_ablation_baseline: bool = True

    def prep_directory(self, test_mode: bool) -> Path:
        """Return where Tutorial 16 writes the volumetric model and its fits."""
        return (
            self.output_directory(test_mode)
            / "tutorial_16_duke_heart_physics_informed_motion"
        )

    def infer_directory(self, test_mode: bool) -> Path:
        """Return where Tutorial 18 writes predictions, scores and USD."""
        return (
            self.output_directory(test_mode)
            / "tutorial_18_duke_heart_physics_informed_motion"
        )

    def ssm_template_file(self, test_mode: bool) -> Path:
        """Return the tetrahedral template Tutorial 16 fills the mean surface with.

        Its cells are the elements the strain energy is summed over, and its
        points are the ones the network predicts a displacement for, which is
        what makes one set of element node ids valid for every subject.
        """
        return self.prep_directory(test_mode) / "ssm_template.vtu"

    def ssm_model_file(self, test_mode: bool) -> Path:
        """Return the volumetric shape model Tutorial 16 writes and 17 reads.

        Distinct from the inherited ``pca_model_file``, which is Tutorial 6's
        surface model: this model's components are ``3 * n`` wide for the *tet*
        mesh's ``n`` points, so the two are not interchangeable.
        """
        return self.prep_directory(test_mode) / "pca_model.json"

    def ssm_mean_volume_file(self, test_mode: bool) -> Path:
        """Return that model's mean tetrahedral mesh."""
        return self.prep_directory(test_mode) / "pca_mean.vtu"

    def ssm_mean_boundary_file(self, test_mode: bool) -> Path:
        """Return the bounding surface of that mean mesh, for display only."""
        return self.prep_directory(test_mode) / "pca_mean_surface.vtp"

    def physics_informed_weights_directory(self, test_mode: bool) -> Path:
        """Return the network Tutorial 17 trains and Tutorial 18 infers with.

        Beside the Tutorial 9 weights rather than inside them: a physics-informed
        model is trained on a different shape model, so it can neither resume
        from nor overwrite that checkpoint.
        """
        return (
            self.weights_directory(test_mode)
            / "physicsnemo_physics_informed_motion_duke_heart"
        )

    def ablation_weights_directory(self, test_mode: bool) -> Path:
        """Return the ``lambda_physics = 0`` model Tutorial 17 trains to compare."""
        return (
            self.weights_directory(test_mode)
            / "physicsnemo_physics_informed_motion_duke_heart_ablation"
        )

    def epochs(self, test_mode: bool) -> int:
        """Return the training epoch count for this run mode."""
        return self.number_of_epochs_test if test_mode else self.number_of_epochs


#: The single instance every physics-informed Duke heart tutorial imports.
DUKE_HEART_PHYSICS_INFORMED = ParametersDukeHeartPhysicsInformed()
