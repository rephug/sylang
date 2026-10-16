# Sylang Documentation

This directory contains comprehensive documentation for the Sylang constructed language and its tokenizer.

## Directory Structure

- [/tokenizer](tokenizer/) - Documentation related to the Sylang tokenizer design and implementation
- [/research](research/) - Research papers and reports on Sylang's performance and benefits

## Tokenizer Documentation

- [Engineering for Token Efficiency](tokenizer/engineering-for-token-efficiency.md) - Detailed explanation of Sylang's design principles and implementation approaches
- [Tokenizer Design](tokenizer/tokenizer-design.md) - Comprehensive guide to the Sylang Prime tokenizer design for Gemma 3 and Qwen 3 models

## Research Papers

- [One Token, One Meaning](research/one-token-one-meaning.md) - How Sylang Prime addresses polysemanticity in language models
- [LLM Performance and Benefits Report](research/Sylang%20Prime_%20LLM%20Performance%20and%20Benefits%20Report.pdf) - Detailed analysis of performance improvements with Sylang
- [Copy bottleneck diagnosis (2026-10-13)](research/copy-bottleneck-diagnosis-2026-10-13.md) - Why the tiny byte model formats answers but does not copy identifiers, with runtime probes and one-factor runs on the project's own model
- [Evidence assessment and plan (2026-10-16)](research/sylang-evidence-assessment-and-plan-2026-10-16.md) - What has and has not been demonstrated (Sylang Prime, the core-v0.1 formats, the SYLANG-BATCH compact format), the machine-efficiency ideas from the M design measured under the Qwen 3 tokenizer, the next experiment on a pretrained model, a US$100 pilot and a staged US$1,000 plan with gates, the record-count-line test, and the history cleanup explained
- [Pretrained-model pilot, revised (2026-10-16)](research/sylang-pilot-revised-2026-10-16.md) - Critical comparison with a second-opinion plan, one revised staged pilot (profile, matched comparison, decision stage) under a US$100 all-in ceiling with pre-registered non-inferiority rules, provider re-check with dollars per completed experiment, owner setup steps, and the reproducible package prepared on the server
- [GPU provider re-check (2026-10-16)](research/gpu-provider-recheck-2026-10-16.md) - Official rates, credits, billing and teardown behaviour of eight GPU options with dollars per completed experiment, and what could not be verified
