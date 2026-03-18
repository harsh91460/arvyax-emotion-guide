# Error Analysis - 10 Key Failure Cases

## ️⃣1 Short Text: "ok"
**Input**: "ok" | **True**: neutral | **Pred**: calm
**Why failed**: No semantic signal
**Fix**: Add length-based uncertainty flag

## ️⃣2 Conflicting Signals  
**Input**: High energy(5)+overwhelmed text
**True**: overwhelmed | **Pred**: focused  
**Why**: Energy overpowered text signal
**Fix**: Weighted ensemble

## ️⃣3 Ambiguous: "fine i guess"
**Input**: "fine i guess" | **True**: neutral | **Pred**: calm
**Why**: Sarcasm missed
**Fix**: Sentiment polarity check

## ️⃣4 Night + High Energy
**Input**: energy=5, time=night | **True**: restless
**Pred**: deep_work (wrong timing)
**Why**: Context ignored
**Fix**: Time-of-day rules

## ️⃣5 Sleep Debt
**Input**: sleep_hours=3.5 | **True**: tired
**Pred**: neutral
**Why**: Threshold too high
**Fix**: sleep_hours < 5 → rest

## ️⃣6 Spelling Errors
**Input**: "teh rain helped" | **True**: calm
**Pred**: overwhelmed  
**Why**: TF-IDF hurt by typos
**Fix**: Spell correction

## ️⃣7 Session Length
**Input**: duration_min=4 | **True**: overwhelmed
**Pred**: calm
**Why**: Short sessions = distraction
**Fix**: duration < 10 → uncertainty++

## ️⃣8 Face Emotion Conflict
**Input**: happyface + overwhelmed text
**Pred**: calm (wrong)
**Why**: Face > text weight
**Fix**: Text primary, face secondary

##️ 9 Morning Overwhelm
**Input**: time=morning, overwhelmed
**Pred**: deep_work
**Why**: Wrong assumption
**Fix**: Morning overwhelmed → breathing

## 10 Rare States
**Input**: "vague" reflection_quality
**Pred**: Always neutral
**Why**: Class imbalance
**Fix**: SMOTE oversampling

## Key Insights
- **Text > Metadata** (52% vs 48% importance)
- **Short texts** = high uncertainty needed
- **Time-of-day** critical for decisions
- **Sleep_hours < 5** = always rest priority
