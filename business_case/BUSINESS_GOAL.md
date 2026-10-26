# Business Goal — Grouped Wearable Activity Recognition

## Product scenario

The proposed product is a consumer wearable wellness assistant that uses
smartwatch accelerometer and gyroscope signals to distinguish between:

- physical activity;
- possible food-or-drink activity;
- other daily activity.

The model operates on five-second sensor windows. A window prediction is
evidence about the wearer's current behavior, not by itself a confirmed
real-world event.

## Intended actions

| Predicted class | Intended product action |
|---|---|
| `physical_activity` | Contribute evidence toward activity and movement tracking |
| `food_or_drink` | Contribute evidence toward a possible intake event or optional logging prompt |
| `other_daily_activity` | Take no targeted action |

A production system should combine consecutive window predictions before
declaring an event or sending a prompt. Event construction and notification
logic are outside the initial window-classification experiment.

## Target definition

| Business label | Original codes | Original activities |
|---|---|---|
| `physical_activity` | A, B, C, M, O, P | Walking, Jogging, Stairs, Kicking Soccer Ball, Playing Catch, Dribbling Basketball |
| `food_or_drink` | H, I, J, K, L | Eating Soup, Eating Chips, Eating Pasta, Drinking from Cup, Eating Sandwich |
| `other_daily_activity` | D, E, F, G, Q, R, S | Sitting, Standing, Typing, Brushing Teeth, Writing, Clapping, Folding Clothes |

`Brushing Teeth` intentionally belongs to `other_daily_activity`. Its hand-to-
mouth motion makes it a valuable negative example for food-or-drink detection.

## Development support

The paired-watch development population contains:

| Business label | Examples | Share |
|---|---:|---:|
| `physical_activity` | 8,470 | 33.11% |
| `food_or_drink` | 7,105 | 27.77% |
| `other_daily_activity` | 10,010 | 39.12% |

All three groups contain all 41 development subjects. This is sufficiently
balanced to begin without class weighting.

## Error costs

A false `food_or_drink` prediction can produce an irrelevant prompt and
notification fatigue. Food-or-drink precision is therefore the principal
business-facing metric.

A missed food-or-drink window can reduce event recall but is less disruptive
than a false notification. Recall must still be reported because a highly
precise model that detects very few intake windows would not be useful.

Physical-activity false positives overstate movement, while false negatives
understate it. Physical-activity recall is therefore an important secondary
metric.

## Evaluation objective

The primary technical comparison metric is mean macro-F1 over the fixed five
subject-wise development folds. This prevents the largest group from dominating
model selection.

Business interpretation must additionally include:

- food-or-drink precision, recall, and F1;
- physical-activity recall;
- the three-class confusion matrix;
- fold variability;
- subject-level performance variability.

## Scope and limitations

The WISDM activities were collected under a controlled activity protocol.
`food_or_drink` predictions indicate sensor patterns associated with the
recorded eating and drinking gestures; they do not prove that food or drink was
actually consumed.

This stage evaluates isolated five-second windows. It does not yet establish:

- event-level meal or drink detection;
- acceptable false notifications per day;
- real-time latency or battery performance;
- performance in uncontrolled daily life;
- medical, nutritional, or caloric conclusions.

Development of the grouped classifier and its business policy remains
restricted to the 41 development subjects and their fixed folds. After the
complete system is frozen, the ten held-out subjects will be evaluated once as
the final test for this three-class target, with no post-test changes. The same
subjects were previously inspected for the separate 18-class task, so they are
unseen for this grouped-target evaluation but not globally unseen within the
repository.
