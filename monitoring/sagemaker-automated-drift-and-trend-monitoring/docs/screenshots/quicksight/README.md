# Understanding Drift Scores: A Visual Guide to ML Model Health

*When your model's world changes, drift scores tell the story*

---

## 🎯 The Story Behind the Numbers

Imagine you trained a fraud detection model in 2023. Your model learned patterns from 10,000 credit card transactions — the average credit limit was $8,000, most transactions happened at grocery stores (merchant category 1000-2000), and customers were typically 3-5 years old accounts.

Fast forward to March 2024. You run 107 new predictions, get the ground truth confirmations, and run your daily drift monitoring. The system compares these 107 new transactions against your original 10,000 training examples. What it finds might surprise you.

**Credit limits have skyrocketed.** What used to cluster around $8,000 is now $50,000-$80,000. Your model sees numbers it never encountered during training. The drift score? **74.47** — a number so high it's practically screaming "something fundamental has changed!"

This is what drift detection is all about: **catching when the real world stops looking like your training data**.

---

## 📊 What is a "Drift Score"?

Think of a drift score as a **distance measurement** between two worlds:
- **World A:** Your training data (the past)
- **World B:** Your new inference data (the present)

The score answers one question: *"How different are these worlds?"*

### The Math (Simplified)

For each feature (like credit_limit, transaction_amount, account_age), the system:

1. **Divides data into buckets** (bins)
   ```
   Training Data:
   $0-5k:     20% of transactions
   $5k-10k:   40% of transactions
   $10k-20k:  25% of transactions
   $20k-50k:  12% of transactions
   $50k+:      3% of transactions
   ```

2. **Checks where new data lands**
   ```
   New Data (107 records):
   $0-5k:      5% of transactions  ← Much lower!
   $5k-10k:    8% of transactions  ← Much lower!
   $10k-20k:  10% of transactions
   $20k-50k:  15% of transactions
   $50k+:     62% of transactions  ← HUGE spike!
   ```

3. **Calculates the difference**
   ```
   PSI (Population Stability Index) formula:
   Σ (Actual% - Expected%) × ln(Actual% / Expected%)

   For credit_limit: 74.47 (extreme drift!)
   For account_age:   5.80 (minimal drift)
   ```

The resulting number tells you **how much the distribution has shifted**.

---

## 🌡️ The Drift Score Scale

Like a fever thermometer for your model:

```
🧊 0.00 - 0.10   HEALTHY
   Feature distribution is stable
   Model sees familiar patterns
   No action needed ✅

🌤️ 0.10 - 0.25   MONITOR
   Moderate changes detected
   Worth investigating
   Consider retraining if trend continues ⚠️

🔥 0.25+          ALERT!
   Significant drift detected
   Model may perform poorly
   Investigate immediately & retrain 🚨
```

**Your credit_limit score of 74.47?** That's not just in the alert zone — it's in the *"the world has fundamentally changed"* zone.

---

## 📈 Reading the Visuals

### Visual 1: Feature Drift Score Timeline

![Feature Drift Timeline Example](DriftScore-Average.png)

This line chart shows **how drift evolves over time** for each feature:

```
Drift Score
  80 |                    🔴 credit_limit (spike!)
     |                   / |
  60 |                  /  |
     |                 /   |
  40 |                /    |
     |               /     |
  20 |    🟢_______/_______|___ distance_from_home
     |   /
   0 |__/________🟡______________ account_age (stable)
     └────────────────────────────► Time
      Mar 21   Mar 22   Mar 23   Mar 24
```

**What each line means:**

- **🟡 Flat line near zero (account_age):**
  - Feature distribution hasn't changed
  - Model still sees familiar values
  - **Action:** None. Everything is fine! ✅

- **🟢 Gentle curve (distance_from_home):**
  - Feature drifting gradually
  - Probably seasonal (more travel in spring?)
  - **Action:** Monitor. May stabilize naturally. ⚠️

- **🔴 Sharp spike (credit_limit):**
  - Sudden, dramatic change
  - Could indicate data quality issue or business change
  - **Action:** Investigate immediately! 🚨

**Key insights from trendlines:**

1. **Pattern Recognition:** Is drift temporary (spike then drop) or permanent (step up and stays)?
2. **Correlation:** Do multiple features drift together? (Suggests systemic change)
3. **Timing:** Did drift start after a deployment? (Possible bug)
4. **Magnitude:** Small bumps are normal. Vertical spikes are red flags.

---

### Visual 2: Top Drifting Features (Bar Chart)

This horizontal bar chart ranks features by **average drift score** across all monitoring runs:

```
Feature                     Avg Drift Score
credit_limit               ████████████████████ 74.47 🔥
merchant_category_code     ████████ 28.25
account_age_days           ██ 5.80
max_transaction_amt        █ 4.17
velocity_score             █ 1.41
```

**What this tells you:**

- **Prioritization:** Start investigating the longest bars first
- **Root cause analysis:** If one feature drifts massively but others don't, it's likely a data quality issue
- **Systemic issues:** If many features drift together, the underlying data distribution has changed

**In your case:**
- `credit_limit` at 74.47 is an **outlier** — likely a data pipeline change or error
- `merchant_category_code` at 28.25 is **moderate** — business patterns may be shifting
- Most other features < 10 are **stable** — model's foundation is solid

---

### Understanding Aggregation Methods: Average vs. Variance-Population

QuickSight provides different aggregation methods for drift scores, each revealing different aspects of feature behavior:

#### drift_score (Average) - The Standard View

![Drift Score Average Aggregation](DriftScore-Sum.png)

**What it shows:**
The **mean drift score** for each feature across all monitoring runs.

**Calculation:**
```
Average = (Run1_Score + Run2_Score + Run3_Score + ... + RunN_Score) / N
```

**Use case:**
- **General health check:** Which features drift most on average?
- **Prioritization:** Where should you focus investigation efforts?
- **Trend identification:** Which features consistently show elevated drift?

**Example:**
If `credit_limit` has drift scores of [70, 75, 78] across 3 runs:
```
Average = (70 + 75 + 78) / 3 = 74.33
```

**Best for:**
- Identifying chronically drifting features
- Understanding typical drift magnitude
- Comparing relative drift across features

---

#### drift_score (Variance - Population) - The Stability View

![Drift Score Variance-Population Aggregation](DriftScore-Variance-Population.png)

**What it shows:**
The **variability/consistency** of drift scores over time for each feature.

**Calculation:**
```
Population Variance = Σ(Score - Mean)² / N

Where:
- Score = drift score for each monitoring run
- Mean = average drift score across all runs
- N = number of monitoring runs
```

**Use case:**
- **Stability analysis:** Which features drift consistently vs. sporadically?
- **Anomaly detection:** Which features have sudden, irregular spikes?
- **Pattern recognition:** Distinguish steady drift from intermittent issues

**Example:**
Two features with same average (30) but different variance:

**Feature A (Consistent drift):**
- Scores: [29, 30, 31, 30, 30]
- Average: 30
- Variance: 0.4 (very stable, **low variance**)
- Interpretation: Feature consistently drifts at ~30

**Feature B (Sporadic drift):**
- Scores: [0, 5, 95, 10, 40]
- Average: 30
- Variance: 1090 (highly variable, **high variance**)
- Interpretation: Feature has unpredictable spikes

**Best for:**
- Detecting data quality issues (high variance = intermittent problems)
- Distinguishing systematic drift from one-off events
- Identifying features that need different monitoring strategies

---

### When to Use Each Aggregation

| Scenario | Use Average | Use Variance-Population |
|----------|-------------|------------------------|
| **Find worst offenders** | ✅ Shows highest average drift | ❌ Doesn't show magnitude |
| **Identify unstable features** | ❌ Masks inconsistency | ✅ Shows erratic behavior |
| **Prioritize retraining** | ✅ High average = chronic issue | ⚠️ Use both together |
| **Detect data quality issues** | ⚠️ May miss intermittent bugs | ✅ High variance = sporadic issues |
| **Monitor seasonal patterns** | ✅ Shows overall impact | ✅ Low variance = predictable |
| **Debug pipeline changes** | ✅ Shows immediate impact | ✅ Variance spike = sudden change |

---

### Real-World Interpretation Example

**Feature: `credit_limit`**

![Credit Limit Example](Feature_drift_time.png)

**Average = 74.47** (very high)
- Tells you: This feature is chronically drifting
- Action: Investigate why values are consistently far from training distribution

**Variance = 150** (moderate)
- Tells you: Drift magnitude varies somewhat across runs
- Action: Check if drift is worsening over time or stabilizing

**Combined interpretation:**
- High average + moderate variance = **Sustained drift with some fluctuation**
- Likely cause: Systematic change (e.g., business rule change, new customer segment)
- Not a data quality issue (would show high variance with lower average)

**Feature: `transaction_amount`**

**Average = 2.15** (low)
- Tells you: Minimal drift on average
- Action: Low priority for investigation

**Variance = 8.50** (high relative to average)
- Tells you: Occasional spikes despite low average
- Action: Monitor for data quality issues or intermittent bugs

**Combined interpretation:**
- Low average + high variance = **Sporadic drift events**
- Likely cause: Occasional data quality issue or edge cases
- Action: Set up alerts for variance spikes, investigate specific runs

---

### Quick Reference: Choosing Your View

**Use Average when:**
- Starting investigation (which features drift most?)
- Prioritizing retraining efforts
- Reporting to stakeholders (simpler to explain)
- Comparing drift across multiple models

**Use Variance-Population when:**
- Debugging data pipeline issues
- Identifying unstable features
- Distinguishing systematic vs. random drift
- Setting up targeted alerts

**Use Both when:**
- Conducting thorough drift analysis
- Designing retraining strategies
- Building automated monitoring rules
- Understanding root causes

---

## 🔍 Real-World Example: Your 107 Records

### What Actually Happened

You ran drift monitoring on **107 new transactions** with confirmed ground truth, comparing them to **10,000 training examples**.

**The Comparison:**

| Feature | Training (2023) | New Data (Mar 2024) | Drift Score | Status |
|---------|----------------|---------------------|-------------|--------|
| **credit_limit** | $5k-$10k average | $50k-$80k average | **74.47** | 🚨 Critical |
| **merchant_category** | 1000-2000 range | 3000-5000 range | **28.25** | ⚠️ High |
| **account_age** | 2-5 years | 2-5 years (same) | **5.80** | ✅ Stable |
| **transaction_amt** | $50-$500 | $60-$550 | **2.15** | ✅ Stable |
| **cvv_match** | 95% match rate | 94% match rate | **0.50** | ✅ Stable |

**What this means:**

1. **Credit limits have exploded** (8x increase)
   - Possible causes:
     - Data pipeline change (wrong column mapped?)
     - Business change (premium card rollout?)
     - Currency conversion error (cents → dollars?)
   - Model impact: **Severe** — model never trained on $50k+ limits

2. **Merchant categories shifted** (new shopping patterns)
   - Possible causes:
     - Seasonal change (tax season, holiday spending)
     - New merchant types (crypto, gig economy)
     - Category code system update
   - Model impact: **Moderate** — model may handle poorly

3. **Most features stable** (low drift)
   - Good news: Core patterns haven't changed
   - Model foundation is still valid
   - Only specific features need attention

---

## 🎬 The Process: From Data to Drift Score

Here's what happens when you click "Run Drift Monitoring":

### Step 1: Data Collection (Automated)
```python
# System queries Athena for recent inferences with ground truth
query = """
SELECT * FROM inference_responses
WHERE ground_truth IS NOT NULL
AND timestamp >= CURRENT_DATE - INTERVAL '7' DAY
"""
# Returns your 107 records
```

### Step 2: Feature Extraction
```python
# System extracts same 29 features used in training
new_features = [
    'transaction_amount',
    'credit_limit',
    'merchant_category_code',
    'account_age_days',
    # ... 25 more features
]
```

### Step 3: Distribution Comparison (Per Feature)
```python
# For EACH of 29 features:
for feature in new_features:
    # Get training distribution
    training_dist = get_buckets(training_data[feature])
    # Example: [20%, 40%, 25%, 12%, 3%]

    # Get new data distribution
    new_dist = get_buckets(new_data[feature])
    # Example: [5%, 8%, 10%, 15%, 62%]

    # Calculate PSI (Population Stability Index)
    drift_score = calculate_psi(training_dist, new_dist)
    # Example: 74.47 for credit_limit
```

### Step 4: Threshold Checking
```python
# Compare against configured thresholds
if drift_score > 0.25:
    severity = "CRITICAL"  # 🔥
elif drift_score > 0.10:
    severity = "WARNING"   # ⚠️
else:
    severity = "OK"        # ✅
```

### Step 5: Alerting & Logging
```python
# If critical drift detected
if drift_score > threshold:
    # Send SNS email alert
    send_alert(f"Feature {feature} drift: {drift_score}")

    # Log to MLflow
    mlflow.log_metric(f"drift_score_{feature}", drift_score)

    # Write to Athena for QuickSight
    write_to_monitoring_responses(feature, drift_score)
```

### Step 6: Visualization Update
```python
# QuickSight queries the new data
# Trendlines update automatically (DIRECT_QUERY mode)
# New data point appears on timeline charts
```

---

## 🧠 What Drift Scores Tell You About Model Health

### Scenario 1: All Features Stable (Drift < 0.10)
```
✅ Model is healthy
✅ Training data still representative
✅ Predictions remain reliable
➡️  Action: Continue monitoring
```

### Scenario 2: One Feature Spikes (Your Case)
```
🔍 Isolated issue detected
🔍 Likely data quality or pipeline problem
🔍 Model may still work for other features
➡️  Action: Investigate specific feature
     Check data pipeline
     Verify data transformations
     Consider feature importance
```

### Scenario 3: Multiple Features Drift Together
```
⚠️  Systemic change in data distribution
⚠️  World has changed since training
⚠️  Model assumptions no longer valid
➡️  Action: Retrain model ASAP
     Use recent data as new training set
     Update feature engineering if needed
```

### Scenario 4: Gradual Upward Trend
```
📈 Slow drift over weeks/months
📈 Adaptation or seasonal effect
📈 Model degrading gradually
➡️  Action: Schedule retrain
     Not urgent but plan ahead
     Consider quarterly retraining schedule
```

---

## 🎯 How to Act on Drift Scores

### For credit_limit (74.47) 🚨

**Immediate Actions:**
1. **Check data pipeline**
   ```sql
   -- Verify recent data looks correct
   SELECT
       MIN(credit_limit) as min,
       MAX(credit_limit) as max,
       AVG(credit_limit) as avg
   FROM inference_responses
   WHERE date >= CURRENT_DATE - 7
   ```

2. **Compare against training**
   ```sql
   -- Should be similar ranges
   SELECT
       MIN(credit_limit),
       MAX(credit_limit),
       AVG(credit_limit)
   FROM training_data
   ```

3. **Look for patterns**
   - Did it change on a specific date? (deployment)
   - Is it all records or a subset? (data quality)
   - Does it correlate with other changes? (system update)

4. **Investigate root cause**
   - Contact data engineering team
   - Review recent pipeline changes
   - Check for unit conversion errors
   - Verify source system hasn't changed

### For merchant_category (28.25) ⚠️

**Monitor and Plan:**
1. **Understand the change**
   ```python
   # Get distribution of categories
   training_cats = training_data['merchant_category'].value_counts()
   new_cats = new_data['merchant_category'].value_counts()

   # Compare
   print("New categories:", set(new_cats.index) - set(training_cats.index))
   ```

2. **Assess impact**
   - Is model accuracy affected?
   - Are false positives increasing?
   - Check confusion matrix for these categories

3. **Consider retraining**
   - If trend continues for 2-3 weeks
   - If model performance degrades
   - Include new merchant categories in training

### For stable features (< 10) ✅

**No Action Needed:**
- These features are working as expected
- Model handles them well
- Continue monitoring for future changes

---

## 📚 Technical Deep Dive: PSI Calculation

For the data scientists who want the math:

**Population Stability Index (PSI) Formula:**

```
PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
```

**Example: credit_limit bins**

| Bin | Training % | New % | Diff | ln(New/Train) | Component |
|-----|-----------|-------|------|---------------|-----------|
| $0-5k | 20% | 5% | -15% | ln(0.25) = -1.39 | 0.208 |
| $5k-10k | 40% | 8% | -32% | ln(0.20) = -1.61 | 0.515 |
| $10k-20k | 25% | 10% | -15% | ln(0.40) = -0.92 | 0.138 |
| $20k-50k | 12% | 15% | +3% | ln(1.25) = 0.22 | 0.007 |
| $50k+ | 3% | 62% | +59% | ln(20.67) = 3.03 | 1.788 |

**Total PSI = 0.208 + 0.515 + 0.138 + 0.007 + 1.788 = 2.656**

(Note: Your actual PSI of 74.47 suggests even more extreme shifts or normalized differently by Evidently)

**Interpretation:**
- PSI < 0.10: No significant change
- PSI 0.10-0.25: Moderate shift
- PSI > 0.25: Significant shift requiring action

---

## 🎓 Key Takeaways

1. **Drift scores measure how much your data distribution has changed** since training

2. **Trendlines show temporal patterns** — spikes vs. gradual changes vs. stability

3. **Bar charts prioritize investigation** — start with highest scores

4. **Context matters** — one spiking feature is different from all features drifting

5. **Action thresholds:**
   - < 10: Monitor 👀
   - 10-25: Investigate 🔍
   - 25+: Act now! 🚨

6. **Drift doesn't always mean retrain** — could be data quality issues

7. **Your model's health depends on** stable feature distributions matching training data

---

## 🔗 Related Documentation

- **Main README:** `/README.md` — Complete system architecture
- **Drift Configuration:** `/src/config/config.yaml` — Adjust thresholds and sensitivity
- **QuickSight Setup:** `/notebooks/3_governance_dashboard.ipynb` — Dashboard creation
- **Feature-Level Drift Visuals:** `FEATURE_LEVEL_SUMMARY.md` — Add detailed per-feature drift analysis to your dashboard

---

## 💡 Pro Tips

**For Data Scientists:**
- Check feature importance: High drift in low-importance features may be acceptable
- Look for correlation: Drifting features that don't affect predictions are less critical
- Consider seasonality: Some drift is expected (holiday shopping, tax season)

**For ML Engineers:**
- Set up automated alerts for PSI > 0.25
- Schedule weekly reviews of drift trends
- Create runbooks for common drift scenarios

**For Business Stakeholders:**
- Drift scores translate to model reliability
- High drift = higher risk of poor predictions
- Regular monitoring prevents silent model failures

---

**Remember:** Drift detection is your early warning system. Like smoke detectors in a building, these scores alert you *before* the fire spreads. A spike in drift today could prevent a model failure tomorrow.

*Keep your models healthy. Monitor your drift scores.* 📊✨
