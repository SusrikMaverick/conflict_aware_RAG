# Methods

All six methods use the same Qwen2.5-7B-Instruct model. They differ in prompts and answer selection; no new model was trained.

## Docs + Cards

Extract up to three cards from each document. Check the quoted text and its source offsets, then give the question, all documents and cards to the model. This is the original method.

## Evidence Pruning

Expand the original answer with coverage proposals that occur in the source text. Build a candidate pool from recovery proposals and earlier drafts. Two audits check role, scope and counterevidence; an answer is removed only when both meet the fixed deletion rule.

## Memory + Cards

Start with a question-only memory draft. Use an earlier answer as a fallible draft. One final request asks the model to work from the documents first and then use the cards as reminders. These are instructions inside one request, not two isolated calls.

## Context Review

Use a memory draft, all documents and an earlier conservative answer. The final prompt checks the question context and conflicting claims. This last step has no cards directly; an earlier draft can depend on cards.

## Answer Agreement

Starting from the same prior and earlier draft, produce one answer from documents and another from documents plus cards. Keep answers present in both lists after case and whitespace normalization. Semantic aliases are not merged.

## Dual Review

Combine coverage proposals with answers from documents, cards and memory to create an audit pool. Run two audits in different candidate orders. Delete a candidate from the coverage answer only when both audits reject it.

## Development

The work started with document cards, then added coverage proposals and candidate checks. Later variants used memory drafts, earlier answers and intersections. The six displayed workflows were chosen after observing the three datasets.

Earlier stages run automatically. Original document text is retained in document-aware steps. Model memory, source quotes and agreement between two calls can all be wrong. The source-card module locates text; it does not run MiniCheck.
