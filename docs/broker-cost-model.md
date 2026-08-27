# Broker cost and spread model

The simulator separates three sources of friction:

1. quoted bid/ask spread;
2. Alpaca commission, normally zero for a self-directed retail API account but configurable;
3. pass-through SEC, FINRA TAF/CAT, ORF, and OCC fees.

For one option contract, the unrounded buy-side regulatory total is $0.0403:
CAT $0.0003 + ORF $0.015 + OCC $0.025. A sale adds those items plus TAF
$0.00329 and the SEC transaction fee based on sale value. Alpaca aggregates
each fee type daily and rounds each resulting total up to the nearest cent, so
the cash charge can exceed the arithmetic per-fill estimate for small counts.

The cost function also models an unchanged-market round trip from ask to bid.
For example, a $0.02 option spread costs $2.00 per contract before regulatory
fees. This makes spread selection much more economically important than the
headline commission.

Rates are versioned in `configs/alpaca_fee_schedule_2026-07-20.json`. They must
be revalidated before the event because regulatory rates may change.
