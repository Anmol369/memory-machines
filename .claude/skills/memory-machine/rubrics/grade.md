# Grading Rubric (T0–T3)

Source: `memory_machines/tiering/instructions/rate_ungrounded.txt` (Kirkby & Matuschak, 2026). Trimmed and reproduced here so the skill is self-contained.

Rate each generated memory prompt on the scale T0–T3. Keep only **T2 and T3**. Discard T0 and T1 silently.

## Scale

**T3 — Ready to Review.** Well-targeted to the user's interest and well-constructed for SRS. Usable as-is. The question is specific, unambiguous, and will consistently trigger the same retrieval pathway. The answer captures a meaningful detail worth remembering long-term.

**T2 — Needs Polish.** Semantically aligned and reviewable now. Would benefit from minor tweaks to wording or framing. Core targeting and construction are sound.

**T1 — Needs Refactor.** Roughly in the right region of meaning but ineffective for review in current form. Common failures: question too vague, lacks cues, too wordy, or lacks depth. Requires substantial restructuring.

**T0 — Off-Target.** Focuses on details not aligned with the user's likely interest. May be well-constructed as a flashcard but targets the wrong information.

## The critical T1/T2 threshold

This is the boundary between reviewable and non-reviewable. Be strict.

- **T2 is reviewable now.** Polish is cosmetic.
- **T1 is NOT reviewable.** Even if semantically aligned, question-side flaws make reliable recall impossible.

## Evaluating the question, not the answer

Apply stringency to the question — the question cues recall. Answers may contain auxiliary context; that's fine. What matters is whether the question reliably points to one specific retrieval target.

Question-quality checks:

- **Specific retrieval target?** Could the question legitimately elicit multiple valid answers? If yes → T1.
- **Sufficient cues?** Does the question include enough context, specificity, or framing to reliably trigger the intended answer? If connection is loose → T1.
- **Not too generic/definitional?** Generic questions like "what is X?" or "what is the key principle of X?" are often T1 even when semantically on-target. Prefer contextual framing: "what advantage does X have over Y?", "in the context of Z, what distinguishes A from B?"
- **Contextual anchors preserved?** If the source used a specific named comparison, scenario, or framing and the question replaces it with a generic reference, that's a T1 loss of specificity.

A prompt can be semantically on-target but still be T1 if the question fails these checks. Don't downgrade just because the answer has extra context.

## Procedure per card

1. **Identify the user's likely interest.** Quote the specific part of the highlight that was likely the point of interest.
2. **Check targeting.** Does the prompt target that interest? If not → T0.
3. **Check construction.** Run the question-quality checks above. If any reveal vagueness, ambiguity, or lost specificity → T1.
4. **Gap analysis.**
   - Minor wording/clarification changes only → T2
   - Core structural changes needed (split compound, change scope, restructure) → T1
   - Wrong focus or wrong predicate → T1
   - Wrong recall target entirely → T0
5. Assign a final tier. Apply the keep/discard rule: **T2 and T3 → keep. T0 and T1 → discard.**

When in doubt between T1 and T2, default to T1. The dataset's research shows forgiving graders produce worse card decks over time — the bar for "reviewable" must stay high.
