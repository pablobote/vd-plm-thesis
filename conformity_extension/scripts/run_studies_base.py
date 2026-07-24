#!/usr/bin/env python3
"""Run the two-round social-pressure study with brief judgments only."""

from __future__ import annotations

from run_studies_variants_common import (
    ROUND1_PROMPT_TEMPLATE,
    ROUND2_PROMPT_TEMPLATE,
    RoundSpec,
    StudyVariantConfig,
    run_from_cli,
)


CONFIG = StudyVariantConfig(
    variant_name="base",
    description=(
        "Run the two-round social-pressure study with before/after judgments "
        "and confidence ratings."
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
    ),
    default_batch_size=64,
    default_max_tokens=32,
)


if __name__ == "__main__":
    run_from_cli(CONFIG)
