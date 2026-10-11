# Same Alpha, Different Assistance

Assistance magnitude was computed as ||v_exec - v_BCI|| for shared-control samples.
The primary analysis excludes explicit safety-override samples plus a 150-ms buffer because those samples intentionally suppress BCI influence.

The collected table contains 275,382 valid shared-control samples from 3,966 trials and 60 sessions.
Safety-override plus pad samples were 14.2% of valid samples and were excluded from the primary distribution summary.
After exclusion, 236,409 samples remained across 16 populated alpha bins of width 0.05.
Across alpha bins, the median within-bin 90th-10th percentile range of assistance magnitude was 1.924 (range 1.024-2.614).
Including safety-override samples gave the same qualitative result: median within-bin 90th-10th percentile range 2.404.

Reviewer-facing summary:

Even within narrow prior-confidence bins, assistance magnitude was broadly distributed. Thus alpha alone does not determine the amount of correction: alpha indexes prior concentration, while the executed correction also depends on the relationship between the decoded BCI command and the controller-favored command.
