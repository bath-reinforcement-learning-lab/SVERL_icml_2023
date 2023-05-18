import numpy as np
from collections import defaultdict
from utils import F_not_i

class Shapley:
    """
    Calculates Shapley values given characteristic values.
    """

    def __init__(self, states_to_explain):
        
        # For Shapley calculations
        self.F_card = len(states_to_explain[0])
        self.F = np.arange(self.F_card)
        self.states = states_to_explain

    def run(self, characteristic_values):
        """
        Calculates all the shapley values for every state and feature.
        """

        shapley_values = defaultdict(lambda: [[] for _ in range(self.F_card)])

        for state in self.states:

            # All characteristic values for a given state.
            C_values = {C: value_table[tuple(state)] for C, value_table in characteristic_values.items()}
            
            for feature in self.F:

                for C in F_not_i(self.F, feature): # All coalitions without feature

                    # Cardinal of C
                    C_card = len(C)

                    # Add our feature to the current coalition
                    C_with_i = np.append(C, feature).astype(int)
                    C_with_i.sort()

                    # Rolling sum, following formula
                    shapley_values[tuple(state)][feature].append(np.math.factorial(C_card) * np.math.factorial(self.F_card - C_card - 1) * (C_values[tuple(C_with_i)] - C_values[tuple(C)]))

                # Final weighting and return
                shapley_values[tuple(state)][feature] = np.sum(shapley_values[tuple(state)][feature], axis=0) / np.math.factorial(self.F_card)

        return dict(shapley_values)
