#!/usr/bin/env python3
"""Run the social-pressure study with an added third-round self-reflection turn."""

from __future__ import annotations

from run_studies_variants_common import (
    ROUND1_PROMPT_TEMPLATE,
    ROUND2_PROMPT_TEMPLATE,
    ROUND3_SELF_REFLECT_TEMPLATE,
    RoundSpec,
    StudyVariantConfig,
    run_from_cli,
)


CONFIG = StudyVariantConfig(
    variant_name="self_reflect",
    description=(
        "Run the social-pressure study with a third round that asks the model "
        "to reflect on the conversation before answering again."
    ),
    round_specs=(
        RoundSpec(
            name="before",
            user_template=ROUND1_PROMPT_TEMPLATE,
        ),
        RoundSpec(
            name="after",
            user_template=ROUND2_PROMPT_TEMPLATE,
        ),
        RoundSpec(
            name="after_reflection",
            user_template=ROUND3_SELF_REFLECT_TEMPLATE,
        ),
    ),
    default_batch_size=32,
    default_max_tokens=128,
)


if __name__ == "__main__":
    run_from_cli(CONFIG)
