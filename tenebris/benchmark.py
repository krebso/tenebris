import json

from collections import defaultdict

from polars import DataFrame
from torch.utils.data import DataLoader
from tqdm import tqdm

from tenebris.domain.interfaces.method import ExplainabilityMethod
from tenebris.domain.interfaces.metric import Metric, ReduceStrategy

REDUCE_STRATEGY_HANDLERS = {
    ReduceStrategy.ACCURACY: lambda l: sum(map(int, l)) / len(l),
    ReduceStrategy.AVERAGE: lambda l: sum(l) / len(l),
}


class BenchmarkService:
    def __init__(self, metrics: list[Metric], methods: list[ExplainabilityMethod]) -> None:
        self._metrics = metrics
        self._methods = methods  # TODO: think about if this makes sense, or should be part of the run

        self._results: dict = {metric.name: defaultdict(list) for metric in self._metrics}

    def _append_results(self, metric: Metric, results: dict) -> None:
        for key, value in results.items():
            self._results[metric.name][key].append(value)

    def run(self, data: DataLoader) -> None:
        for d in tqdm(data):
            input_, target = d
            for metric in self._metrics:
                result = metric.compute(methods=self._methods, input_=input_, target=target)
                self._append_results(metric, result)

    def results(self) -> dict:
        return self._results

    def save_results(self, path: str) -> None:
        json.dump(self._results, open(path, "w"))

    def load_results(self, path: str) -> None:
        self._results = json.load(open(path))

    def reduce_results(self) -> dict:
        # TODO: we want some stat here as well, for average the percentiles as well
        return {
            metric.name: {
                method.name: REDUCE_STRATEGY_HANDLERS[metric.reduce_strategy](self._results[metric.name][method.name])
                for method in self._methods
            }
            for metric in self._metrics
        }

    def get_result_df(self) -> DataFrame:
        result = self.reduce_results()
        df = DataFrame(
            {
                "Benchmark": list(result.keys()),
                **{method.name: [values[method.name] for values in result.values()] for method in self._methods},
            }
        )
        return df
