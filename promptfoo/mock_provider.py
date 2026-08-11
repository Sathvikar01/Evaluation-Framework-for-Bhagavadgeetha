"""Offline provider used only to verify the Promptfoo integration."""


def call_api(prompt, options, context):
    if "action" in str(prompt).lower():
        ref = "BhG 2.47"
    else:
        ref = "BhG 18.66"
    return {
        "output": f"The answer is grounded in {ref}.",
        "metadata": {"retrieved_refs": [ref]},
    }

