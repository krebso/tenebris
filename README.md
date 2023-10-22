# Requirements on the benchmarking
- speed of prediction
- usability of predictions - how well they assess what the network is doing
- Using chosen out of the co-12 methods


# Premise
- We will benchmark only the positive tiles and slides cancer detection is the
very same thing as anomaly detection. We attribute negativity simply to the
absence of positive features and not to any particular trait of the input.
Therefore, explaining why the model decided against can be simply reduced to the fact
that it did not find relevant features.
- We benchmark tiles, not wsi. Our model is trained to predict cancer in tile, and the explanations are generated per tile. This means, that the explanations are local to the tile and do not consider its context within the whole slide image. While combining (averaging) explanations over the while wsi produces heatmap that can be nicely put over the original image, it does not capture the attributions in their actual strengt, as we are not able to retrospectively reconstruct precise values for them.
- When saving explanations for the predictions, we will save tensor of the attribuions generated for individual tiles. This, when sampling the input wsi, allows us to quickly access true explanation for the prediction, without having to recompute them.

# Roadmap
- Able to run mlflow experiment :white_check_mark:
- Able to run one benchmark for one method, configurable via hydra :white_check_mark:
- Able to run multiple benchmarks for one method, configurable via hydra :x:
- Able to run multiple benchmarks for multiple methods, configurable via hydra :x:

# Benchmarking pipeline flow
1. Orchestrator downlads all mlflow artifacts and initializes all neccessary objects
2. Given a dataloader, we iterate over tuples in form of [input, attribution, explanation], and we feed those parallely to the benchmarks
3. Each benchmark gets model and this tuple, computes the relevant metric and updates its internal state as neccessary
4. After dataloader ends, each benchmark logs its metric


# Methods to use to benchmark
- deletion check
- alignment with domain knowledge (EHR)
- alignment with the old occlusion results (Ask matej to provide them if possible)

1. Correctness
   - Model Parameter Randomization Check:<br>
   Localization: Randomly perturb the internals of the predictive model and check that the explanation changes.
   - Controlled synthetic Data Check:<br>
   Controlled experiment: Create a synthetic dataset such that the predictive model
   should follow a particular reasoning, known a priori (important: checking this
   assumption by, e.g., reporting almost-perfect accuracy). Evaluate whether the
   explanation shows the same reasoning as the data generation process.
   - Single Deletion:<br>
   Delete, mask or perturb a single feature in the input and evaluate
   the change in output of the predictive model. Measure correlation with explanation’s importance score.
   - Incremental Deletion (or Incremental Addition):<br>
   One by one delete (or perturb) or add features to the input, based on explanation’s
   order, and measure for each new input the change in output of the predictive model.
   Report average change in log-odds score, AUC, steepness of curve or number of
   features needed for a different decision. Compare with random ranking or other baselines. 
2. Completeness
   - Preservation Check:<br>
   Giving the explanation (or data based on the explanation) as input to the predictive
   model should result in the same decision as for the original, full input sample.
   - Deletion Check<br>
   Giving input without explanation’s relevant features should result in a different
   decision by the predictive model than the decision for the original, full input sample.
3. Contrastivity
   - Target Sensitivity<br>
   The explanation for a particular target or model output (e.g., class) should be different
   from an explanation for another target.
4. Covariate Complexity:
   - Covariate Homogeneity
   Evaluate how consistently a covariate (i.e., feature) in an explanation represents a
   predefined human-interpretable concept.
   - Covariate Regularity<br>
   Evaluate the regularity of an explanation by measuring its Shannon entropy, to
   quantify how noisy the explanation is and how easy it is to memorize the explanation.
5. Confidence
   - Confidence Accuracy<br>
   Measure the accuracy of confidence/uncertainty estimates if these are present in the
   explanation.
     - While this was not essentially meant to be computed for heatmaps, I present a way in which
     this technique can be adapted to our domain. Given we have an explanation annotated by experts, we
     can look at the confidence of explanation of single feature given by the value of attributions and
     whether the attributed feature is annotated by pathologist as well. We take into account that the annotations
     are not pixel level precise, and we allow (yada yada random number based on the width of the pen) the attribution
     to be little off
6. Coherence
   - Alignment with Domain Knowledge:<br>
   Compare the generated explanation with a “ground-truth” expected explanation based on domain knowledge.
   - XAI Methods Agreement:<br>
   Quantitatively compare explanations from different XAI methods and evaluate their agreement.
     - Can be used as comparison with occlusion
     - I can probably compute it pairwise for all methods as well just because I can
       - There was an article which was comparing (and somewhat proving) similarity of some of the gradient
       attributing methods, I can quote this as well.
7. Human Feedback Impact
   - Controllability
   Measure the improvement of explanation quality after human feedback, where the user is seen as a system component.
  

# MVP
- load model from mlflow uri
- load artifacts from mlflow uri
- be able to output tuple of tiles (tile_from_wsi, annotated_heatmap_over_tile, explanation_heatmap)
  - this should actually be a 3 separate dataloaders initialized with the same config - enabling to have the same coordinates of tiles
  - with sane design this should be easily achievable as the dataloader will be just interested in WSI? Probably not cause the annotations, but let's see
    - there we can create a dataloader that will create 0-1 tensor according to the annotation the two remaining can be the same

# MVP Breakdown


# Ideas
- be able to benchmark the method itself
    - may make sense to create task for explaining model as well instead of the retarded callbacks - at least locally
- Check the torch-script (numba equivalent) for the JIT compilation


# Notes
- we are not evaluating the function f computed by the model
- we evaluate the explainable function e and this is what needs to be considered when designing the methods
- explainability is non-binary characteristic


# Caveats
- in the co 12, some of the incremental adding / deleting may be same as just having the positively and negatively
explained areas ran through the model, be careful when doing this

# Co by sa dalo robit dalej...
- v Co-12 clanku boli niektore metriky ktore fungovali len za pouzitia viac modelov, dalo by sa pozriet na to


# Hydra notes
- We have 3 configuration files. 
- hydra.yaml - constructed from the base configs inside .yaml files
- overrides.yaml - command line overrides
- config.yaml - this is the final one, apllied after resolving the hydra.yaml and applying overrides
- When I want to run the previous experiment, I can do it like this `python experiment.py --config-dir=outputs/... --config-name=config.yaml`
- omegaconf is hydra configuration manager
- When I have the DictConfig object, I can access the parameters via `.` or as in dictionary or as in `glom`.
- Composing multiple conf files
- `instantiate` this will allow me to initialize the methods whoohoo

# OmegaConf notes
- Variable interpolation: 
    ```
    foo: "Hello"
    bar: "World"
    message: "${foo} ${bar}!"  # "Hello World!"
    ```
- Resolver - Variable interpolation using custom function, too complicated, try to avoid that