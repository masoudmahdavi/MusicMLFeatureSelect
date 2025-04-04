from typing import Protocol


class FeatureSelectionStrategy(Protocol):
    def select_feature(self) -> str:
        pass

class Func1Selector(FeatureSelectionStrategy):
    def select_feature(self) -> str:
        return "Feature selected using Method 1"

class Func2Selector(FeatureSelectionStrategy):
    def select_feature(self) -> str:
        return "Feature selected using Method 2"
    
class FeatureSelectionContext:
    def __init__(self, strategy: FeatureSelectionStrategy):
        self.strategy = strategy

    def execute_selection(self):
        return self.strategy.select_feature()

# Example usage
method = input("Choose feature selection method (func1/func2): ").strip()

strategy = Func1Selector() if method == "func1" else Func2Selector()
context = FeatureSelectionContext(strategy)
print(context.execute_selection())