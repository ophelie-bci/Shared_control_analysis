# Neural Target Cross-condition Generalization

Within each session, target classifiers were trained on cue-window MUA firing rates from one condition and tested on the other condition. The feature window was targetOnset to 800 ms after target onset, capped at obstacle appearance when available.

## Session-level decoding

- Appearing Obstacle, BCI-only -> BCI-only: mean accuracy 49.8%, mean majority-class chance 21.8%, above chance 28.0 pp; 19/19 sessions above chance; one-sided Wilcoxon p=1.907e-06.
- Appearing Obstacle, BCI-only -> Shared-control: mean accuracy 48.5%, mean majority-class chance 22.3%, above chance 26.2 pp; 18/19 sessions above chance; one-sided Wilcoxon p=9.788e-05.
- Appearing Obstacle, Shared-control -> BCI-only: mean accuracy 46.1%, mean majority-class chance 21.8%, above chance 24.3 pp; 18/19 sessions above chance; one-sided Wilcoxon p=9.822e-05.
- Appearing Obstacle, Shared-control -> Shared-control: mean accuracy 46.1%, mean majority-class chance 22.3%, above chance 23.8 pp; 19/19 sessions above chance; one-sided Wilcoxon p=1.907e-06.
- Appearing Obstacle follow-up, BCI-only -> BCI-only: mean accuracy 60.1%, mean majority-class chance 25.7%, above chance 34.4 pp; 9/9 sessions above chance; one-sided Wilcoxon p=0.001953.
- Appearing Obstacle follow-up, BCI-only -> Shared-control: mean accuracy 60.8%, mean majority-class chance 28.2%, above chance 32.6 pp; 9/9 sessions above chance; one-sided Wilcoxon p=0.001953.
- Appearing Obstacle follow-up, Shared-control -> BCI-only: mean accuracy 56.3%, mean majority-class chance 25.7%, above chance 30.5 pp; 9/9 sessions above chance; one-sided Wilcoxon p=0.001953.
- Appearing Obstacle follow-up, Shared-control -> Shared-control: mean accuracy 54.4%, mean majority-class chance 28.2%, above chance 26.2 pp; 9/9 sessions above chance; one-sided Wilcoxon p=0.001953.
- Fixed Obstacle, BCI-only -> BCI-only: mean accuracy 47.9%, mean majority-class chance 22.1%, above chance 25.7 pp; 23/23 sessions above chance; one-sided Wilcoxon p=1.192e-07.
- Fixed Obstacle, BCI-only -> Shared-control: mean accuracy 48.7%, mean majority-class chance 22.1%, above chance 26.5 pp; 23/23 sessions above chance; one-sided Wilcoxon p=1.192e-07.
- Fixed Obstacle, Shared-control -> BCI-only: mean accuracy 48.1%, mean majority-class chance 22.1%, above chance 26.0 pp; 23/23 sessions above chance; one-sided Wilcoxon p=1.192e-07.
- Fixed Obstacle, Shared-control -> Shared-control: mean accuracy 49.5%, mean majority-class chance 22.1%, above chance 27.3 pp; 23/23 sessions above chance; one-sided Wilcoxon p=1.192e-07.

## Cross-condition versus within-condition decoding

- Appearing Obstacle, trained on BCI-only: cross-condition accuracy 48.5% vs within-condition accuracy 49.8%; cross-minus-within, chance-corrected = -1.8 pp (median -1.2 pp; two-sided Wilcoxon p=0.5949).
- Appearing Obstacle, trained on Shared-control: cross-condition accuracy 46.1% vs within-condition accuracy 46.1%; cross-minus-within, chance-corrected = 0.5 pp (median 2.1 pp; two-sided Wilcoxon p=0.6794).
- Appearing Obstacle follow-up, trained on BCI-only: cross-condition accuracy 60.8% vs within-condition accuracy 60.1%; cross-minus-within, chance-corrected = -1.8 pp (median -2.0 pp; two-sided Wilcoxon p=0.6523).
- Appearing Obstacle follow-up, trained on Shared-control: cross-condition accuracy 56.3% vs within-condition accuracy 54.4%; cross-minus-within, chance-corrected = 4.3 pp (median 5.8 pp; two-sided Wilcoxon p=0.25).
- Fixed Obstacle, trained on BCI-only: cross-condition accuracy 48.7% vs within-condition accuracy 47.9%; cross-minus-within, chance-corrected = 0.8 pp (median 3.6 pp; two-sided Wilcoxon p=0.482).
- Fixed Obstacle, trained on Shared-control: cross-condition accuracy 48.1% vs within-condition accuracy 49.5%; cross-minus-within, chance-corrected = -1.4 pp (median -3.4 pp; two-sided Wilcoxon p=0.6434).

## Neural target decoding by decoded-choice state

- Appearing Obstacle, BCI-only, correct: mean accuracy 62.1%, mean majority-class chance 50.9%, above chance 11.2 pp; 9/13 sessions above chance; one-sided Wilcoxon p=0.07897.
- Appearing Obstacle, Shared-control, correct: mean accuracy 49.8%, mean majority-class chance 45.2%, above chance 4.7 pp; 9/18 sessions above chance; one-sided Wilcoxon p=0.128.
- Appearing Obstacle, BCI-only, adjacent: mean accuracy 39.5%, mean majority-class chance 44.1%, above chance -4.5 pp; 5/18 sessions above chance; one-sided Wilcoxon p=0.7357.
- Appearing Obstacle, Shared-control, adjacent: mean accuracy 41.5%, mean majority-class chance 35.0%, above chance 6.5 pp; 13/19 sessions above chance; one-sided Wilcoxon p=0.03051.
- Appearing Obstacle, BCI-only, ambiguous: mean accuracy 48.0%, mean majority-class chance 44.5%, above chance 3.5 pp; 6/17 sessions above chance; one-sided Wilcoxon p=0.3246.
- Appearing Obstacle, Shared-control, ambiguous: mean accuracy 54.2%, mean majority-class chance 48.9%, above chance 5.3 pp; 8/16 sessions above chance; one-sided Wilcoxon p=0.1047.
- Appearing Obstacle, Shared-control, incorrect: mean accuracy 40.0%, mean majority-class chance 60.0%, above chance -20.0 pp; 0/1 sessions above chance; one-sided Wilcoxon p=1.
- Appearing Obstacle follow-up, BCI-only, correct: mean accuracy 65.3%, mean majority-class chance 47.9%, above chance 17.4 pp; 9/9 sessions above chance; one-sided Wilcoxon p=0.001953.
- Appearing Obstacle follow-up, Shared-control, correct: mean accuracy 70.1%, mean majority-class chance 41.4%, above chance 28.7 pp; 9/9 sessions above chance; one-sided Wilcoxon p=0.001953.
- Appearing Obstacle follow-up, BCI-only, adjacent: mean accuracy 58.2%, mean majority-class chance 39.7%, above chance 18.5 pp; 9/9 sessions above chance; one-sided Wilcoxon p=0.001953.
- Appearing Obstacle follow-up, Shared-control, adjacent: mean accuracy 46.9%, mean majority-class chance 39.7%, above chance 7.2 pp; 5/9 sessions above chance; one-sided Wilcoxon p=0.125.
- Appearing Obstacle follow-up, BCI-only, ambiguous: mean accuracy 50.9%, mean majority-class chance 43.0%, above chance 7.8 pp; 4/9 sessions above chance; one-sided Wilcoxon p=0.1552.
- Appearing Obstacle follow-up, Shared-control, ambiguous: mean accuracy 52.8%, mean majority-class chance 44.0%, above chance 8.9 pp; 3/7 sessions above chance; one-sided Wilcoxon p=0.2315.
- Appearing Obstacle follow-up, BCI-only, incorrect: mean accuracy 23.8%, mean majority-class chance 46.4%, above chance -22.6 pp; 0/2 sessions above chance; one-sided Wilcoxon p=1.
- Appearing Obstacle follow-up, Shared-control, incorrect: mean accuracy 26.8%, mean majority-class chance 60.7%, above chance -33.9 pp; 0/2 sessions above chance; one-sided Wilcoxon p=1.
- Fixed Obstacle, BCI-only, correct: mean accuracy 60.6%, mean majority-class chance 43.8%, above chance 16.8 pp; 18/23 sessions above chance; one-sided Wilcoxon p=0.0001605.
- Fixed Obstacle, Shared-control, correct: mean accuracy 51.6%, mean majority-class chance 38.5%, above chance 13.2 pp; 18/23 sessions above chance; one-sided Wilcoxon p=0.0007721.
- Fixed Obstacle, BCI-only, adjacent: mean accuracy 46.8%, mean majority-class chance 34.2%, above chance 12.6 pp; 18/23 sessions above chance; one-sided Wilcoxon p=9.332e-05.
- Fixed Obstacle, Shared-control, adjacent: mean accuracy 41.9%, mean majority-class chance 34.5%, above chance 7.4 pp; 14/23 sessions above chance; one-sided Wilcoxon p=0.009766.
- Fixed Obstacle, BCI-only, ambiguous: mean accuracy 41.1%, mean majority-class chance 40.1%, above chance 1.0 pp; 10/22 sessions above chance; one-sided Wilcoxon p=0.444.
- Fixed Obstacle, Shared-control, ambiguous: mean accuracy 47.0%, mean majority-class chance 45.6%, above chance 1.4 pp; 8/17 sessions above chance; one-sided Wilcoxon p=0.3767.
- Fixed Obstacle, BCI-only, incorrect: mean accuracy 60.0%, mean majority-class chance 40.0%, above chance 20.0 pp; 1/1 sessions above chance; one-sided Wilcoxon p=0.5.
- Fixed Obstacle, Shared-control, incorrect: mean accuracy 72.2%, mean majority-class chance 55.6%, above chance 16.7 pp; 1/3 sessions above chance; one-sided Wilcoxon p=0.1587.

## BCI-only versus shared-control within decoded-choice state

- Appearing Obstacle, correct: shared-minus-BCI chance-corrected accuracy = -7.0 pp (median -25.3 pp; n=13; two-sided Wilcoxon p=0.5417).
- Appearing Obstacle, adjacent: shared-minus-BCI chance-corrected accuracy = 12.0 pp (median 7.8 pp; n=18; two-sided Wilcoxon p=0.05994).
- Appearing Obstacle, ambiguous: shared-minus-BCI chance-corrected accuracy = 2.7 pp (median 3.2 pp; n=15; two-sided Wilcoxon p=0.6387).
- Appearing Obstacle follow-up, correct: shared-minus-BCI chance-corrected accuracy = 11.4 pp (median 11.1 pp; n=9; two-sided Wilcoxon p=0.05469).
- Appearing Obstacle follow-up, adjacent: shared-minus-BCI chance-corrected accuracy = -11.3 pp (median -1.4 pp; n=9; two-sided Wilcoxon p=0.25).
- Appearing Obstacle follow-up, ambiguous: shared-minus-BCI chance-corrected accuracy = 9.3 pp (median -4.4 pp; n=7; two-sided Wilcoxon p=0.5781).
- Appearing Obstacle follow-up, incorrect: shared-minus-BCI chance-corrected accuracy = -11.3 pp (median -11.3 pp; n=2; two-sided Wilcoxon p=0.5).
- Fixed Obstacle, correct: shared-minus-BCI chance-corrected accuracy = -3.6 pp (median -4.8 pp; n=23; two-sided Wilcoxon p=0.2595).
- Fixed Obstacle, adjacent: shared-minus-BCI chance-corrected accuracy = -5.2 pp (median -5.8 pp; n=23; two-sided Wilcoxon p=0.09175).
- Fixed Obstacle, ambiguous: shared-minus-BCI chance-corrected accuracy = 0.9 pp (median 0.0 pp; n=17; two-sided Wilcoxon p=0.4603).
- Fixed Obstacle, incorrect: shared-minus-BCI chance-corrected accuracy = -20.0 pp (median -20.0 pp; n=1; two-sided Wilcoxon p=1).

## Suggested reviewer wording

A neural target classifier decoded the cued target from early cue-window population activity, before trial outcome could dominate the signal. Target information generalized across BCI-only and shared-control trials and was close to within-condition held-out decoding. When trials were split by the decoded-choice state of the BCI velocity command, cue-target information was strongest for decoder-correct and adjacent trials. Ambiguous-trial decoding was modest and not significant across sessions, and incorrect-choice trials were too sparse for a reliable session-level inference. Across matched decoder-choice states with sufficient paired sessions, BCI-only and shared-control decoding did not differ significantly, supporting broadly comparable target-related neural representation across conditions.
