from collections import defaultdict

from torch.utils.data import DataLoader

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

        self._results = defaultdict(list)

    def _append_results(self, results: dict) -> None:
        for key, value in results.items():
            self._results[key].append(value)

    def run(self, data: DataLoader) -> None:
        for d in data:
            for metric in self._metrics:
                result = metric.compute(self._methods, **d)
                self._append_results(result)

    def results(self) -> dict:
        return self._results

    def reduced_results(self) -> dict:
        return {
            metric.name: REDUCE_STRATEGY_HANDLERS[metric.reduce_strategy](self._results[metric.name])
            for metric in self._metrics
        }



