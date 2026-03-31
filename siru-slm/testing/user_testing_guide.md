# User Testing Guide (Assistant Directors + Indie Writers)

## Goal
Validate whether rewritten output feels like authentic Tamil cinema dialogue and is useful in real writing workflows.

## Participants
- 5-10 assistant directors
- 5-10 indie writers

## Session Structure (20-30 mins each)
1. Provide the participant with 10 raw dialogue lines.
2. Ask them to rewrite 5 lines manually first.
3. Then let them use Siru SLM modes (`mass`, `emotion`, `subtext`) for all 10 lines.
4. Collect ratings and qualitative feedback.

## Questions
1. Does this output feel like Tamil cinema dialogue? (1-5)
2. Would you use this in your script iteration flow? (1-5)
3. Which mode gives the best value for your work?
4. What felt fake/generic in the output?
5. Which line felt most “theatre whistle-worthy”?

## Capture Format
Store results in `testing/user_feedback.jsonl` with:
- participant_role
- prompt
- mode
- rating_authenticity
- rating_utility
- comments

## Success Criteria
- Avg authenticity >= 4.0
- Avg utility >= 3.8
- >= 60% users say they would reuse the feature
