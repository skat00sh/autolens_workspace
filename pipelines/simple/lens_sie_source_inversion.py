import autofit as af
from autolens.data.array import mask as msk
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline.phase import phase_imaging
from autolens.pipeline import pipeline
from autolens.pipeline import pipeline_tagging
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg

import os

# In this pipeline, we'll perform a basic analysis which fits two source galaxies using a light profile
# followed by an inversion, where the lens galaxy's light is not present in the image, using two phases:

# Phase 1:

# Description: initialize the lens mass model and source light profile.
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Previous Pipelines: None
# Prior Passing: None
# Notes: None

# Phase 2:

# Description: initialize the inversion's pixelization and regularization hyper-parameters, using a previous lens mass
#              model.
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: VoronoiMagnification + Constant
# Previous Pipelines: initialize/lens_sie_source_sersic_from_init.py
# Prior Passing: Lens Mass (variable -> phase 1).
# Notes: None

# Phase 3:

# Description: Refine the lens mass model and source inversion.
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: VoronoiMagnification + Constant
# Previous Pipelines: initialize/lens_sie_source_sersic_from_init.py
# Prior Passing: Lens Mass (variable -> phase 1), source inversion (variable -> phase 2).
# Notes: None

# ***NOTE*** Performing this analysis in a pipeline composed of 3 consectutive phases it not ideal, and it is better to
#            breaking the pipeline down into multiple pipelines. This is what is done in the 'pipelines/no_lens_light'
#            folder, using the pipelines:

#            1) initialize/lens_sie_source_sersic_from_init.py (phases 1->3)
#            2) initialize/lens_sie_source_inversion_from_pipeline.py (phases 4->5)

#            See runners/runner_adding_pipelines.py for more details on adding pipelines.


def make_pipeline(
    pixelization=pix.VoronoiMagnification,
    regularization=reg.Constant,
    phase_folders=None,
    tag_phases=True,
    redshift_lens=0.5,
    redshift_source=1.0,
    sub_grid_size=2,
    signal_to_noise_limit=None,
    bin_up_factor=None,
    positions_threshold=None,
    inner_mask_radii=None,
    interp_pixel_scale=None,
    use_inversion_border=True,
    inversion_pixel_limit=None,
    cluster_pixel_scale=0.1,
):

    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    # We setup the pipeline name using the tagging module. In this case, the pipeline name is not given a tag and
    # will be the string specified below However, its good practise to use the 'tag.' function below, incase
    # a pipeline does use customized tag names.

    pipeline_name = "pl__sie_source_inversion"

    pipeline_name = pipeline_tagging.pipeline_name_from_name_and_settings(
        pipeline_name=pipeline_name,
        pixelization=pixelization,
        regularization=regularization,
    )

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/phase_folder_1/phase_folder_2/pipeline_name/phase_name/settings_tag'

    phase_folders.append(pipeline_name)

    # As there is no lens light component, we can use an annular mask throughout this pipeline which removes the
    # central regions of the image.

    def mask_function_annular(image):
        return msk.Mask.circular_annular(
            shape=image.shape,
            pixel_scale=image.pixel_scale,
            inner_radius_arcsec=0.2,
            outer_radius_arcsec=3.3,
        )

    ### PHASE 1 ###

    # In phase 1, we will fit the lens galaxy's mass and one source galaxy, where we:

    # 1) Set our priors on the lens galaxy (y,x) centre such that we assume the image is centred around the lens galaxy.

    class LensSourceX1Phase(phase_imaging.PhaseImaging):
        def pass_priors(self, results):

            self.galaxies.lens.mass.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
            self.galaxies.lens.mass.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)

    phase1 = LensSourceX1Phase(
        phase_name="phase_1_source",
        phase_folders=phase_folders,
        tag_phases=tag_phases,
        galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=redshift_lens,
                mass=mp.EllipticalIsothermal,
                shear=mp.ExternalShear,
            ),
            source=gm.GalaxyModel(redshift=redshift_source, light=lp.EllipticalSersic),
        ),
        mask_function=mask_function_annular,
        sub_grid_size=sub_grid_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        positions_threshold=positions_threshold,
        inner_mask_radii=inner_mask_radii,
        use_inversion_border=use_inversion_border,
        interp_pixel_scale=interp_pixel_scale,
        optimizer_class=af.MultiNest,
    )

    # You'll see these lines throughout all of the example pipelines. They are used to make MultiNest sample the \
    # non-linear parameter space faster (if you haven't already, checkout 'tutorial_7_multinest_black_magic' in
    # 'howtolens/chapter_2_lens_modeling'.

    # Fitting the lens galaxy and source galaxy from uninitialized priors often risks MultiNest getting stuck in a
    # local maxima, especially for the image in this example which actually has two source galaxies. Therefore, whilst
    # I will continue to use constant efficiency mode to ensure fast run time, I've upped the number of live points
    # and decreased the sampling efficiency from the usual values to ensure the non-linear search is robust.

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 80
    phase1.optimizer.sampling_efficiency = 0.2

    ### PHASE 1 ###

    # In phase 1, we initialize the inversion's resolution and regularization coefficient, where we:

    # 1) Fix our mass model to the lens galaxy mass-model from phase 3 of the initialize pipeline.
    # 2) Use a circular mask which includes all of the source-galaxy light.

    class InversionPhase(phase_imaging.PhaseImaging):
        def pass_priors(self, results):

            ## Lens Mass, SIE -> SIE ###

            self.galaxies.lens.mass = results.from_phase(
                "phase_1_source"
            ).constant.galaxies.lens.mass

            ## Lens Mass, Shear -> Shear ###

            self.galaxies.lens.shear = results.from_phase(
                "phase_1_source"
            ).constant.galaxies.lens.shear

    phase2 = InversionPhase(
        phase_name="phase_2_initialize_inversion",
        phase_folders=phase_folders,
        tag_phases=tag_phases,
        galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=redshift_lens,
                mass=mp.EllipticalIsothermal,
                shear=mp.ExternalShear,
            ),
            source=gm.GalaxyModel(
                redshift=redshift_source,
                pixelization=pixelization,
                regularization=regularization,
            ),
        ),
        sub_grid_size=sub_grid_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        positions_threshold=positions_threshold,
        inner_mask_radii=inner_mask_radii,
        interp_pixel_scale=interp_pixel_scale,
        use_inversion_border=use_inversion_border,
        inversion_pixel_limit=inversion_pixel_limit,
        cluster_pixel_scale=cluster_pixel_scale,
        optimizer_class=af.MultiNest,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 20
    phase2.optimizer.sampling_efficiency = 0.8

    phase2 = phase2.extend_with_inversion_phase()

    ### PHASE 3 ###

    # In phase 3, we fit the lens's mass and source galaxy using an inversion, where we:

    # 1) Initialize the priors on the lens galaxy mass using the results of the previous pipeline.
    # 2) Initialize the priors of all source inversion parameters from phase 1.

    class InversionPhase(phase_imaging.PhaseImaging):
        def pass_priors(self, results):

            ## Lens Mass, SIE -> SIE ###

            self.galaxies.lens.mass = results.from_phase(
                "phase_1_source"
            ).variable.galaxies.lens.mass

            ## Lens Mass, Shear -> Shear ###

            self.galaxies.lens.shear = results.from_phase(
                "phase_1_source"
            ).variable.galaxies.lens.shear

            ### Source Inversion, Inv -> Inv ###

            self.galaxies.source = results.from_phase(
                "phase_2_initialize_inversion"
            ).inversion.constant.galaxies.source

    phase3 = InversionPhase(
        phase_name="phase_3_inversion",
        phase_folders=phase_folders,
        tag_phases=tag_phases,
        galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=redshift_lens,
                mass=mp.EllipticalIsothermal,
                shear=mp.ExternalShear,
            ),
            source=gm.GalaxyModel(
                redshift=redshift_source,
                pixelization=pixelization,
                regularization=regularization,
            ),
        ),
        sub_grid_size=sub_grid_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        positions_threshold=positions_threshold,
        inner_mask_radii=inner_mask_radii,
        interp_pixel_scale=interp_pixel_scale,
        use_inversion_border=use_inversion_border,
        inversion_pixel_limit=inversion_pixel_limit,
        cluster_pixel_scale=cluster_pixel_scale,
        optimizer_class=af.MultiNest,
    )

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 50
    phase3.optimizer.sampling_efficiency = 0.5

    return pipeline.PipelineImaging(pipeline_name, phase1, phase2, phase3)
