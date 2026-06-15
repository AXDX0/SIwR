from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


class CartPoleBayesNet:

    def __init__(self):

        self.model = DiscreteBayesianNetwork([
            ("PoleAngle", "Action"),
            ("PoleVelocity", "Action")
        ])

        cpd_angle = TabularCPD(
            variable="PoleAngle",
            variable_card=3,
            values=[
                [1/3],
                [1/3],
                [1/3]
            ],
            state_names={
                "PoleAngle": ["Left", "Center", "Right"]
            }
        )

        cpd_velocity = TabularCPD(
            variable="PoleVelocity",
            variable_card=3,
            values=[
                [1/3],
                [1/3],
                [1/3]
            ],
            state_names={
                "PoleVelocity": [
                    "FallingLeft",
                    "Stable",
                    "FallingRight"
                ]
            }
        )

        cpd_action = TabularCPD(
            variable="Action",
            variable_card=2,
            evidence=["PoleAngle", "PoleVelocity"],
            evidence_card=[3, 3],

            values=[
                [
                    0.95, 0.85, 0.70,
                    0.85, 0.50, 0.15,
                    0.30, 0.10, 0.05
                ],
                [
                    0.05, 0.15, 0.30,
                    0.15, 0.50, 0.85,
                    0.70, 0.90, 0.95
                ]
            ],

            state_names={
                "Action": ["Left", "Right"],
                "PoleAngle": [
                    "Left",
                    "Center",
                    "Right"
                ],
                "PoleVelocity": [
                    "FallingLeft",
                    "Stable",
                    "FallingRight"
                ]
            }
        )

        self.model.add_cpds(
            cpd_angle,
            cpd_velocity,
            cpd_action
        )

        self.model.check_model()

        self.infer = VariableElimination(self.model)

    def get_distribution(
        self,
        angle_state,
        velocity_state
    ):

        result = self.infer.query(
            variables=["Action"],
            evidence={
                "PoleAngle": angle_state,
                "PoleVelocity": velocity_state
            }
        )

        return result.values