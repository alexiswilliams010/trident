"""Explicit direct and indirect relationships between exact endpoints."""

from copy import deepcopy


def relationship_questions(questions):
    updated = deepcopy(questions)
    updated["topology"] = {
        "type": "choice",
        "instructions": (
            "What relationship does the evidence establish between the two exact selected endpoints? "
            "Distinguish a direct connection from behavior reached through an intermediate operation."
        ),
        "criteria": {
            "forward": (
                "The source directly acts on the exact selected target. The target itself is the recipient, "
                "not merely an operation inside the recipient or downstream of it."
            ),
            "reverse": (
                "The target directly acts on the exact selected source. The source itself is the recipient, "
                "not merely an operation inside the recipient or downstream of it."
            ),
            "indirect": (
                "The endpoints are related through an intermediate operation. For example, the source "
                "calls the target's enclosing function, which can execute the selected target expression. "
                "This establishes an indirect relationship rather than a direct edge between the selected endpoints."
            ),
            "shared_resource": "Both exact endpoints interact with the same evidenced resource.",
            "none": "The evidence does not establish a direct, indirect, or shared-resource relationship between the exact endpoints.",
        },
    }
    return updated


def recipient_questions(questions):
    result = deepcopy(questions)
    result["condition"] = {
        "type": "choice",
        "instructions": "What does the source's binding or destination identify relative to the exact selected target?",
        "criteria": {
            "exact_target": "The exact selected target itself.",
            "intermediate": "An enclosing function or another intermediate operation, rather than the exact selected target.",
            "shared_resource": "A shared resource with which both selected endpoints interact.",
            "different_destination": "A different destination, with no established intermediate or shared-resource relationship to the selected target.",
            "unresolved": "Not established by the evidence.",
        },
    }
    return result
