import pickle

from pymatgen.util.testing import PymatgenTest

from lematerial_forgebench.preprocess.base import PreprocessorResult
from lematerial_forgebench.preprocess.distribution_preprocess import (
    DistributionPreprocessor,
)
from lematerial_forgebench.preprocess.universal_stability_preprocess import (
    UniversalStabilityPreprocessor,
)
from lematerial_forgebench.benchmarks.distribution_benchmark import DistributionBenchmark
from lematerial_forgebench.benchmarks.diversity_benchmark import DiversityBenchmark
from lematerial_forgebench.benchmarks.stability_benchmark import StabilityBenchmark
from lematerial_forgebench.benchmarks.validity_benchmark import ValidityBenchmark


# dataset loading - reference data is needed for distribution benchmarks. Currently a sample 
# of LeMatBulk
with open("data/full_reference_df.pkl", "rb") as f:
    test_lemat = pickle.load(f)


# From some input model, sample structures. Will replace the pmg test development statement below
test = PymatgenTest()
structures = [
    test.get_structure("Si"),
    test.get_structure("Si"),
    test.get_structure("Si"),

    test.get_structure("LiFePO4"),
]

# run preprocessors on structure samples

# distribution preprocessor
distribution_preprocessor = DistributionPreprocessor()
dist_preprocessor_result = distribution_preprocessor(structures)
timeout = 60
output_dfs = {}

# TODO 
# currently this runs just one MLIP, but the preprocessors will shortly be able to run
# multiple MLIPs internally and aggregate these resutls

# stability preprocessor using mace 
mlip = "mace"
stability_preprocessor = UniversalStabilityPreprocessor(
    model_name=mlip, relax_structures=True, timeout=timeout
)
stability_preprocessor_result = stability_preprocessor(structures)

# combine stability and distribution preprocessor into overall preprocessor results
final_processed_structures = []

for ind in range(0, len(dist_preprocessor_result.processed_structures)):
    combined_structure = dist_preprocessor_result.processed_structures[ind]
    for entry in stability_preprocessor_result.processed_structures[
        ind
    ].properties.keys():
        combined_structure.properties[entry] = (
            stability_preprocessor_result.processed_structures[
                ind
            ].properties[entry]
        )
    final_processed_structures.append(combined_structure)

preprocessor_result = PreprocessorResult(
    processed_structures=final_processed_structures,
    config={
        "stability_preprocessor_config": stability_preprocessor_result.config,
        "distribution_preprocessor_config": dist_preprocessor_result.config,
    },
    computation_time={
        "stability_preprocessor_computation_time": stability_preprocessor_result.computation_time,
        "distribution_preprocessor_computation_time": dist_preprocessor_result.computation_time,
    },
    n_input_structures=stability_preprocessor_result.n_input_structures,
    failed_indices={
        "stability_preprocessor_failed_indices": stability_preprocessor_result.failed_indices,
        "distribution_preprocessor_failed_indices": dist_preprocessor_result.failed_indices,
    },
    warnings={
        "stability_preprocessor_warnings": stability_preprocessor_result.warnings,
        "distribution_preprocessor_warnings": dist_preprocessor_result.warnings,
    },
)

# run benchmarks 

# validity benchmark 
# benchmark = ValidityBenchmark()
# benchmark_result = benchmark.evaluate(structures)

# print("valid_ratio")
# print(benchmark_result.final_scores["valid_structures_ratio"])

# print("charge_neutrality")
# print(
#     benchmark_result.evaluator_results["charge_neutrality"]["metric_results"][
#         "charge_neutrality"
#     ].metrics
# )

# print("interatomic_distance")
# print(
#     benchmark_result.evaluator_results["interatomic_distance"]["metric_results"][
#         "min_distance"
#     ].metrics
# )
# print("physical_plausibility")
# print(
#     benchmark_result.evaluator_results["physical_plausibility"]["metric_results"][
#         "plausibility"
#     ].metrics
# )
# print("overall_validity")
# print(
#     benchmark_result.evaluator_results["overall_validity"]["metric_results"][
#         "composite"
#     ].metrics
# )


# stability benchmark 
benchmark = StabilityBenchmark()
benchmark_result = benchmark.evaluate(
    preprocessor_result.processed_structures
)


print("stable_ratio")
print(
    benchmark_result.evaluator_results["stability"]["metric_results"][
        "stability"
    ].metrics
)
print("metastable_ratio")
print(
    benchmark_result.evaluator_results["metastability"]["metric_results"][
        "metastability"
    ].metrics
)
print("mean_e_above_hull")
print(
    benchmark_result.evaluator_results["mean_e_above_hull"]["metric_results"][
        "mean_e_above_hull"
    ].metrics
)
print("mean_formation_energy")
print(
    benchmark_result.evaluator_results["formation_energy"]["metric_results"][
        "formation_energy"
    ].metrics
)
print("mean_relaxation_RMSE")
print(
    benchmark_result.evaluator_results["relaxation_stability"]["metric_results"][
        "relaxation_stability"
    ].metrics
)
# diversity benchmark 
benchmark = DiversityBenchmark()
benchmark_result = benchmark.evaluate(structures)

print("element_diversity")
print(
    benchmark_result.evaluator_results["element_diversity"]["metric_results"][
        "element_diversity"
    ].metrics
)
print("space_group_diversity")
print(
    benchmark_result.evaluator_results["space_group_diversity"]["metric_results"][
        "space_group_diversity"
    ].metrics
)
print("site_number_diversity")
print(
    benchmark_result.evaluator_results["site_number_diversity"]["metric_results"][
        "site_number_diversity"
    ].metrics
)
print("physical_size_diversity")
print(
    benchmark_result.evaluator_results["physical_size_diversity"]["metric_results"][
        "physical_size_diversity"
    ].metrics
)


# # distribution benchmark 
# benchmark = DistributionBenchmark(reference_df=test_lemat)
# benchmark_result = benchmark.evaluate(
#     preprocessor_result.processed_structures
# )

# print("JSDistance")
# print(
#     benchmark_result.evaluator_results["JSDistance"]["metric_results"][
#         "JSDistance"
#     ].metrics
# )
# print(
#     "Average JSDistance: "
#     + str(
#         benchmark_result.evaluator_results["JSDistance"]["JSDistance_value"]
#     )
# )
# print("MMD")
# print(
#     benchmark_result.evaluator_results["MMD"]["metric_results"][
#         "MMD"
#     ].metrics
# )
# print(
#     "Average MMD: "
#     + str(benchmark_result.evaluator_results["MMD"]["MMD_value"])
# )
# print(mlip + " FrechetDistance")
# print(
#     benchmark_result.evaluator_results["FrechetDistance"]["metric_results"][
#         "FrechetDistance"
#     ].metrics
# )
# print(
#     "Average Frechet Distance: "
#     + str(
#         benchmark_result.evaluator_results["FrechetDistance"][
#             "FrechetDistance_value"
#         ]
#     )
# )

# # HHI benchmark 

# # novelty benchmark 

# # uniqueness benchmark 

# # SUN Benchmark 
# pass # TODO implement 


