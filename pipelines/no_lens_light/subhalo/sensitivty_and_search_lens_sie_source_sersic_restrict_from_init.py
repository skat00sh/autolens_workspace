from autofit.tools import path_util
from autofit.optimize import non_linear as nl
from autofit.mapper import prior
from autofit.tools import phase as autofit_ph
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp

import os

# In this pipeline, we'll perform a subhalo analysis which determines the sensitivity map of a strong lens and 
# then attempts to detection subhalos by putting subhalos at fixed intevals on a 2D (y,x) grid. The pipeline is as 
# follows:

# Phase 1:

# Description: Perform the sensitivity analysis for subhalo locations.
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: AdaptiveMagnification + Constant
# Subhalo: SphericalTruncatedNFWChallenge
# Previous Pipelines: initializers/lens_sie_source_sersic_from_init.py
# Prior Passing: Lens mass, source light (constant -> previous pipeline).
# Notes: Uses a 3D grid of subhalo (y,x) and mass, which is set via the config.

# Phase 2:

# Description: Perform the subhalo detection analysis.
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: AdaptiveMagnification + Constant
# Subhalo: SphericalTruncatedNFWChallenge
# Previous Pipelines: initializers/lens_sie_source_sersic_from_init.py
# Prior Passing: Lens mass (constant -> previous pipeline), source light (variable -> previous pipeline).
# Notes: Priors on subhalo are tuned to give realistic masses (10^6 - 10^10) and concentrations (6-24)

# Phase 3:

# Description: Refine the best-fit detected subhalo from the previous phase, by varying also the lens mass model..
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: AdaptiveMagnification + Constant
# Subhalo: SphericalTruncatedNFWChallenge
# Previous Pipelines: initializers/lens_sie_source_sersic_from_init.py
# Prior Passing: Lens mass (variable -> previous pipeline), source light and subhalo mass (variable -> phase 2).
# Notes: None

# Phase 4:

# Description: Change SIE mass profile to PowerLaw, to refine power-law slope.
# Lens Mass: EllipitcalPowerLaw + ExternalShear
# Source Light: AdaptiveMagnification + Constant
# Subhalo: SphericalTruncatedNFWChallenge
# Previous Pipelines: None
# Prior Passing: Lens mass, source light and subhalo mass (variable -> phase 3).
# Notes: Uses an interpolation pixel scale for fast power-law deflection angle calculations.

def make_pipeline(phase_folders=None, interp_pixel_scale=0.05):

    pipeline_name = 'pipeline_subhalo_sensitivity_and_search_lens_sie_source_sersic_restrict'

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/phase_folder_1/phase_folder_2/pipeline_name/phase_name/'
    phase_folders = path_util.phase_folders_from_phase_folders_and_pipeline_name(phase_folders=phase_folders,
                                                                                pipeline_name=pipeline_name)

    # ### Phase 1 ###
    #
    # # In phase 1, we perform the sensitivity analysis of our lens, using a grid search of subhalo (y,x) coordinates and
    # # mass, where:
    #
    # # 1) The lens model and source-pixelization parameters are held fixed to the best-fit values from phase 2.
    #
    # class GridPhase(ph.LensSourcePlanePhase):
    #
    #     def pass_priors(self, results):
    #
    #         self.lens_galaxies.lens.mass = results.from_phase('phase_1_source').constant.lens.mass
    #         self.source_galaxies.source = results.from_phase('phase_1_source').constant.source
    #
    #         self.lens_galaxies.subhalo.mass.centre_0 = prior.UniformPrior(lower_limit=-2.0, upper_limit=2.0)
    #         self.lens_galaxies.subhalo.mass.centre_1 = prior.UniformPrior(lower_limit=-2.0, upper_limit=2.0)
    #         self.lens_galaxies.subhalo.mass.kappa_s = prior.UniformPrior(lower_limit=0.00001, upper_limit=0.002)
    #         self.lens_galaxies.subhalo.mass.scale_radius = 5.0
    #
    # phase2 = GridPhase(phase_name='phase_2_sensitivity', phase_folders=phase_folders,
    #                    lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal,
    #                                                           shear=mp.ExternalShear),
    #                                       subhalo=gm.GalaxyModel(mass=mp.SphericalNFW)),
    #                    source_galaxies=dict(source=gm.GalaxyModel(light=lp.EllipticalSersic)),
    #                    optimizer_class=nl.GridSearch)

    ### Phase 2 ###

    # In phase 2, we attempt to detect subhalos, by performing a NxN grid search of MultiNest searches, where:

    # 1) The lens model and source-pixelization parameters are held fixed to the best-fit values from phase 1 of the 
    #    initialization pipeline.
    # 2) Each grid search varies the subhalo (y,x) coordinates and mass as free parameters.
    # 3) The priors on these (y,x) coordinates are UniformPriors, with limits corresponding to the grid-cells.

    class GridPhase(autofit_ph.as_grid_search(ph.LensSourcePlanePhase)):

        @property
        def grid_priors(self):
            return [self.variable.subhalo.mass.centre_0, self.variable.subhalo.mass.centre_1]

        def pass_priors(self, results):

            self.lens_galaxies.lens.mass = results.from_phase('phase_1_source').constant.lens.mass
            self.lens_galaxies.lens.shear = results.from_phase('phase_1_source').constant.lens.shear
            
            self.lens_galaxies.subhalo.mass.kappa_s = prior.UniformPrior(lower_limit=0.0001, upper_limit=0.1)
            self.lens_galaxies.subhalo.mass.scale_radius = prior.UniformPrior(lower_limit=0.0, upper_limit=5.0)
            self.lens_galaxies.subhalo.mass.centre_0 = prior.UniformPrior(lower_limit=-2.0, upper_limit=2.0)
            self.lens_galaxies.subhalo.mass.centre_1 = prior.UniformPrior(lower_limit=-2.0, upper_limit=2.0)
            
            centre_mean = results.from_phase('phase_1_source').constant.source.light.centre
            
            self.source_galaxies.source.light.centre.centre_0 = prior.GaussianPrior(mean=centre_mean[0],  sigma=0.5)
            self.source_galaxies.source.light.centre.centre_1 = prior.GaussianPrior(mean=centre_mean[1],  sigma=0.5)
            self.source_galaxies.source.light.intensity = results.from_phase('phase_1_source').variable.source.light.intensity
            self.source_galaxies.source.light.effective_radius = results.from_phase('phase_1_source').constant.source.light.effective_radius
            self.source_galaxies.source.light.sersic_index = results.from_phase('phase_1_source').constant.source.light.sersic_index
            self.source_galaxies.source.light.axis_ratio = results.from_phase('phase_1_source').constant.source.light.axis_ratio
            self.source_galaxies.source.light.phi = results.from_phase('phase_1_source').constant.source.light.phi

    phase2 = GridPhase(phase_name='phase_2_subhalo_search', phase_folders=phase_folders,
                       lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal,
                                                              shear=mp.ExternalShear),
                                          subhalo=gm.GalaxyModel(mass=mp.SphericalTruncatedNFWChallenge)),
                       source_galaxies=dict(source=gm.GalaxyModel(light=lp.EllipticalSersic)),
                       number_of_steps=4, optimizer_class=nl.MultiNest)

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.5

    class SubhaloPhase(ph.LensSourcePlanePhase):

        def pass_priors(self, results):

            self.lens_galaxies.lens.mass = results.from_phase('phase_1_source').variable.lens.mass
            self.lens_galaxies.lens.shear = results.from_phase('phase_1_source').variable.lens.shear
            self.lens_galaxies.subhalo.mass = results.from_phase('phase_2_subhalo_search').best_result.variable.subhalo.mass
            self.source_galaxies.source = results.from_phase('phase_2_subhalo_search').best_result.variable.source

    phase3 = SubhaloPhase(phase_name='phase_3_subhalo_refine', phase_folders=phase_folders,
                          lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal,
                                                                 shear=mp.ExternalShear),
                                             subhalo=gm.GalaxyModel(mass=mp.SphericalTruncatedNFWChallenge)),
                          source_galaxies=dict(source=gm.GalaxyModel(light=lp.EllipticalSersic)),
                          optimizer_class=nl.MultiNest)

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 80
    phase3.optimizer.sampling_efficiency = 0.3

    class SubhaloPhase(ph.LensSourcePlanePhase):

        def pass_priors(self, results):

            self.lens_galaxies.lens.shear = results.from_phase('phase_3_subhalo_refine').variable.lens.shear
            self.lens_galaxies.subhalo.mass = results.from_phase('phase_3_subhalo_refine').variable.subhalo.mass
            self.source_galaxies.source = results.from_phase('phase_3_subhalo_refine').variable.source

            self.lens_galaxies.lens.mass.centre_0 = results.from_phase('phase_3_subhalo_refine').variable.lens.mass.centre_0
            self.lens_galaxies.lens.mass.centre_1 = results.from_phase('phase_3_subhalo_refine').variable.lens.mass.centre_1
            self.lens_galaxies.lens.mass.axis_ratio = results.from_phase('phase_3_subhalo_refine').variable.lens.mass.axis_ratio
            self.lens_galaxies.lens.mass.phi = results.from_phase('phase_3_subhalo_refine').variable.lens.mass.phi

            einstein_radius_mean = results.from_phase('phase_3_subhalo_refine').constant.lens.mass.einstein_radius
            self.lens_galaxies.lens.mass.einstein_radius = prior.GaussianPrior(mean=einstein_radius_mean,  sigma=0.2)

    phase4 = SubhaloPhase(phase_name='phase_4_power_law', phase_folders=phase_folders,
                          lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalPowerLaw,
                                                                 shear=mp.ExternalShear),
                                             subhalo=gm.GalaxyModel(mass=mp.SphericalTruncatedNFWChallenge)),
                          source_galaxies=dict(source=gm.GalaxyModel(light=lp.EllipticalSersic)),
                          interp_pixel_scale=interp_pixel_scale,
                          optimizer_class=nl.MultiNest)

    phase4.optimizer.const_efficiency_mode = True
    phase4.optimizer.n_live_points = 80
    phase4.optimizer.sampling_efficiency = 0.3

    return pipeline.PipelineImaging(pipeline_name, phase2, phase3, phase4)