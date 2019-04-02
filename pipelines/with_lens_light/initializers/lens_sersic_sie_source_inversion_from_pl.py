from autofit.tools import path_util
from autofit.optimize import non_linear as nl
from autofit.mapper import prior
from autolens.data.array import mask as msk
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg

# In this pipeline, we'll continue from the pipeline above to initialize the source galaxy fit using an adaptive
# inversion. This pipeline uses two phases:

# Phase 1:

# Description: Initializes the source inversion parameters.
# Lens Light: EllipticalSersic
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: AdaptiveMagnification + Constant
# Previous Pipelines: initializers/lens_sie_source_sersic_from_init.py
# Prior Passing: Lens Mass (variable -> previous pipeline).
# Notes: None

# Phase 2:

# Description: Refines the lens light and mass models using the source inversion.
# Lens Light: EllipticalSersic
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: AdaptiveMagnification + Constant
# Previous Pipelines: initializers/lens_sie_source_sersic_from_init.py
# Prior Passing: Lens Mass (variable -> previous pipeline), Source Inversion (constant -> phase 1)
# Notes: None

def make_pipeline(phase_folders=None, positions_threshold=None):

    pipeline_name = 'pipeline_init_lens_sersic_sie_source_inversion'

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/phase_folder_1/phase_folder_2/pipeline_name/phase_name/'
    phase_folders = path_util.phase_folders_from_phase_folders_and_pipeline_name(phase_folders=phase_folders,
                                                                                pipeline_name=pipeline_name)


    ### PHASE 1 ###

    # In phase 1, we initialize the inversion's resolution and regularization coefficient, where we:

    # 1) Use a lens-subtracted image generated by subtracting model lens galaxy image from phase 1 of the initializer
    #    pipeline.
    # 2) Fix our mass model to the lens galaxy mass-model from phase 2 of the initializer pipeline.
    # 3) Use a circular mask which includes all of the source-galaxy light.

    class InversionPhase(ph.LensSourcePlanePhase):

        def modify_image(self, image, results):
            return image - results.from_phase('phase_3_lens_light_mass_and_source_light').unmasked_lens_plane_model_image

        def pass_priors(self, results):

            self.lens_galaxies.lens.mass = results.from_phase('phase_3_lens_light_mass_and_source_light').constant.lens.mass
            self.lens_galaxies.lens.shear = results.from_phase('phase_3_lens_light_mass_and_source_light').constant.lens.shear

    phase1 = InversionPhase(phase_name='phase_1_inversion_init', phase_folders=phase_folders,
                            lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal,
                                                                   shear=mp.ExternalShear)),
                            source_galaxies=dict(source=gm.GalaxyModel(pixelization=pix.AdaptiveMagnification,
                                                                      regularization=reg.Constant)),
                            optimizer_class=nl.MultiNest)

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 20
    phase1.optimizer.sampling_efficiency = 0.8

    ### PHASE 2 ###

    # In phase 2, we fit the len galaxy light, mass and source galaxy simultaneously, using an inversion. We will:

    # 1) Initialize the priors of the lens galaxy and source galaxy from phase 3 of the previous pipeline and phase 1
    #    of this pipeline.
    # 2) Use a circular mask including both the lens and source galaxy light.

    class InversionPhase(ph.LensSourcePlanePhase):

        def modify_image(self, image, results):
            return image - results.from_phase('phase_3_lens_light_mass_and_source_light').unmasked_lens_plane_model_image

        def pass_priors(self, results):

            self.lens_galaxies.lens.mass = results.from_phase('phase_3_lens_light_mass_and_source_light').variable.lens.mass
            self.lens_galaxies.lens.shear = results.from_phase('phase_3_lens_light_mass_and_source_light').variable.lens.shear
            self.source_galaxies.source = results.from_phase('phase_1_inversion_init').variable.source

    phase2 = InversionPhase(phase_name='phase_2_inversion', phase_folders=phase_folders,
                            lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal,
                                                                   shear=mp.ExternalShear)),
                            source_galaxies=dict(source=gm.GalaxyModel(pixelization=pix.AdaptiveMagnification,
                                                                      regularization=reg.Constant)),
                            positions_threshold=positions_threshold,
                            optimizer_class=nl.MultiNest)

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 75
    phase2.optimizer.sampling_efficiency = 0.2

    return pipeline.PipelineImaging(pipeline_name, phase1, phase2)