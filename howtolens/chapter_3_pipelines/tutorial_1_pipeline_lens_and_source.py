import autofit as af
import autolens as al

# All pipelines begin with a comment describing the pipeline and a phase-by-phase description of what it does.

# In this pipeline, we'll perform a basic analysis which fits a source galaxy using a parametric light profile and a
# lens galaxy where its light is included and fitted, using three phases:

# Phase 1) Fit the lens galaxy's light using an elliptical Sersic light profile.

# Phase 2) Use this lens subtracted image to fit the lens galaxy's mass (SIE) and source galaxy's light (Sersic).

# Phase 3) Fit the lens's light, mass and source's light simultaneously using priors initialized from the above 2 phases.


def make_pipeline(phase_folders=None):

    # Pipelines takes 'phase_folders' as input, which in conjunction with the pipeline name specify the path structure of
    # the output. In the pipeline runner we pass the phase_folders ['howtolens, c3_t1_lens_and_source], which means the
    # output of this pipeline go to the folder 'autolens_workspace/output/howtolens/c3_t1_lens_and_source/pipeline__light_and_source'.

    # By default, the pipeline folders is None, meaning the output go to the directory 'output/pipeline_name',
    # which in this case would be 'output/pipeline_light_and_source'.

    # In the example pipelines found in 'autolens_workspace/pipelines' folder, we pass the name of our strong lens dataset
    # to the pipeline path. This allows us to fit a large sample of lenses using one pipeline and store all of their
    # results in an ordered directory structure.

    pipeline_name = "pipeline__light_and_source"

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/phase_folder_1/phase_folder_2/pipeline_name/pipeline_tag/phase_name/phase_tag/'
    phase_folders.append(pipeline_name)

    ### PHASE 1 ###

    # First, we create the phase, using the same notation we learnt before (noting the masks function is passed to
    # this phase ensuring the anti-annular masks above is used).

    phase1 = al.PhaseImaging(
        phase_name="phase_1__lens_sersic",
        phase_folders=phase_folders,
        galaxies=dict(lens=al.GalaxyModel(redshift=0.5, light=al.lp.EllipticalSersic)),
        optimizer_class=af.MultiNest,
    )

    # We'll use the MultiNest black magic we covered in tutorial 7 of chapter 2 to get this phase to run fast.

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 30
    phase1.optimizer.sampling_efficiency = 0.5
    phase1.optimizer.evidence_tolerance = 100.0

    ### PHASE 2 ###

    # In phase 2, we fit the source galaxy's light. Thus, we want to make 2 changes from the previous phase.

    # 1) We want to fit the lens subtracted image calculated in phase 1, instead of the observed image.

    # To modify an image, we call a new function, 'modify image'. This function behaves like the pass-priors functions
    # before, whereby we create a python 'class' in a Phase to set it up.  This ensures it has access to the pipeline's
    # 'results' (which you may have noticed was in the the customize_priors functions as well).

    # To setup the modified image we take the observed image and subtract-off the model image from the
    # previous phase, which, if you're keeping track, is an image of the lens galaxy. However, if we just used the
    # 'model_image' in the fit, this would only include pixels that were masked. We want to subtract the lens off the
    # entire image - fortunately, PyAutoLens automatically generates an 'unmasked_lens_plane_model_image' as well!

    class LensSubtractedPhase(al.PhaseImaging):
        def modify_image(self, image, results):
            phase_1_results = results.from_phase("phase_1__lens_sersic")
            return image - phase_1_results.unmasked_model_visibilities_of_planes[0]

    # The function above demonstrates the most important thing about pipelines - that every phase has access to the
    # results of all previous phases. This means we can feed information through the pipeline and therefore use the
    # results of previous phases to setup new phases.

    # You should see that this is done by using the phase_name of the phase we're interested in, which in the above
    # code is named 'phase_1__lens_sersic' (you can check this on line 73 above).

    # We'll do this again in phase 3 and throughout all of the pipelines in this chapter and the autolens_workspace examples.

    # We setup phase 2 as per usual. Note that we don't need to pass the modify image function.

    phase2 = LensSubtractedPhase(
        phase_name="phase_2__lens_sie__source_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(redshift=0.5, mass=al.mp.EllipticalIsothermal),
            source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
        ),
        optimizer_class=af.MultiNest,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.3
    phase2.optimizer.evidence_tolerance = 100.0

    ### PHASE 3 ###

    # Finally, in phase 3, we want to fit the lens and source simultaneously.

    # We'll use the 'customize_priors' function that we all know and love to do this. However, we're going to use the
    # 'results' argument that, in chapter 2, we ignored. This stores the results of the lens model of
    # phases 1 and 2 meaning we can use it to initialize phase 3's priors!

    lens = al.GalaxyModel(
        redshift=0.5, light=al.lp.EllipticalSersic, mass=al.mp.EllipticalIsothermal
    )
    source = al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic)

    # To link two priors together we invoke the 'model' attribute of the previous results. By invoking
    # 'model', this means that:

    # 1) The parameter will be a free-parameter fitted for by the non-linear search.
    # 2) It will use a GaussianPrior based on the previous results as its initialization (we'll cover how this
    #    Gaussian is setup in tutorial 4, for now just imagine it links the results in a sensible way).

    # We can simply link every source galaxy parameter to its phase 2 inferred value, as follows

    source.light.centre_0 = phase2.result.model.galaxies.source.light.centre_0

    source.light.centre_1 = phase2.result.model.galaxies.source.light.centre_1

    source.light.axis_ratio = phase2.result.model.galaxies.source.light.axis_ratio

    source.light.phi = phase2.result.model.galaxies.source.light.phi

    source.light.intensity = phase2.result.model.galaxies.source.light.intensity

    source.light.effective_radius = (
        phase2.result.model.galaxies.source.light.effective_radius
    )

    source.light.sersic_index = phase2.result.model.galaxies.source.light.sersic_index

    # However, listing every parameter like this is ugly and becomes cumbersome if we have a lot of parameters.

    # If, like in the above example, you are making all of the parameters of a lens or source galaxy variable,
    # you can simply set the source galaxy equal to one another without specifying each parameter of every
    # light and mass profile.

    source = (
        phase2.result.model.galaxies.source
    )  # This is identical to lines 196-203 above.

    # For the lens galaxies we have a slightly weird circumstance where the light profiles requires the
    # results of phase 1 and the mass profile the results of phase 2. When passing these as a 'model', we
    # can split them as follows

    lens.light = phase1.result.model.galaxies.lens.light
    lens.mass = phase2.result.model.galaxies.lens.mass

    phase3 = al.PhaseImaging(
        phase_name="phase_3__lens_sersic_sie__source_sersic",
        phase_folders=phase_folders,
        galaxies=dict(lens=lens, source=source),
        optimizer_class=af.MultiNest,
    )

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 50
    phase3.optimizer.sampling_efficiency = 0.3

    return al.PipelineDataset(pipeline_name, phase1, phase2, phase3)
