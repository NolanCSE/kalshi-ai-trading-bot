# Ideology Agent Documentation

The Ideology Agent is a sophisticated addition to the Kalshi AI Trading Bot that applies your worldview, beliefs, and domain expertise to prediction market analysis. Unlike other agents that rely solely on news and reasoning, the Ideology Agent retrieves relevant theory from your personal knowledge library and interprets current events through your ideological framework.

## Overview

### What It Does

The Ideology Agent:
1. **Retrieves relevant knowledge** from your curated document library using RAG (Retrieval-Augmented Generation)
2. **Interprets current news** through your worldview lens
3. **Infers confidence** from the strength of your belief statements
4. **Generates predictions** that may diverge from consensus
5. **Triggers manual review** when your worldview significantly differs from the ensemble

### When It Applies

The Ideology Agent only applies to markets in certain categories:
- **Applicable**: politics, economics, technology, climate, regulation, policy
- **Excluded**: sports, weather, entertainment, celebrities, pure chance

Markets are automatically filtered - the agent returns neutral predictions for excluded categories.

## Setup

### 1. Install Dependencies

```bash
pip install faiss-cpu pymupdf
```

Or if you already installed the updated requirements:
```bash
pip install -r requirements.txt
```

### 2. Create Your Worldview Configuration

Copy the template and customize:

```bash
cp config/worldview.yaml config/my_worldview.yaml
```

Edit `config/worldview.yaml` to reflect your actual beliefs and expertise.

### 3. Build Your Knowledge Library

Create a `library/` directory with your documents:

```
library/
├── economic_theory/
│   ├── friedman_capitalism_and_freedom.pdf
│   └── your_notes_on_monetarism.md
├── tech_prediction/
│   ├── kurzweil_singularity.pdf
│   └── your_ai_timeline_notes.md
├── political_theory/
│   └── your_political_framework.txt
└── personal_notes/
    └── your_prediction_framework.md
```

Supported formats: PDF, Markdown (.md), Text (.txt)

### 4. Index Your Documents

The knowledge library will automatically index documents on first run. To force re-indexing:

```python
from src.utils.knowledge_library import get_knowledge_library

kl = await get_knowledge_library()
await kl.refresh_index()
```

## Configuration Reference

### Worldview Structure

```yaml
ideology_agent:
  enabled: true
  model: "anthropic/claude-sonnet-4.5"
  
  # When to trigger manual review
  override_triggers:
    min_divergence_from_ensemble: 0.20  # >20% different from other agents
    min_worldview_confidence: 0.70       # You have >70% confidence
    min_divergence_from_market: 0.15     # Market price differs by >15%
    max_position_size_pct: 0.05          # Position >5% of portfolio

worldview:
  political:
    framework: "techno_libertarian"      # Name your framework
    confidence_in_framework: 0.75        # How sure are you?
    key_beliefs:
      - "Free markets outperform regulation in tech"
      - "AI development accelerates faster than expected"
  
  economic:
    framework: "monetarist_with_deflation"
    confidence_in_framework: 0.70
    key_beliefs:
      - "Tech deflation keeps inflation lower than Fed predicts"
      
  technology:
    framework: "exponential_growth_realist"
    confidence_in_framework: 0.80
    key_beliefs:
      - "LLM capabilities are consistently underestimated"

domain_expertise:
  - area: "artificial_intelligence"
    expertise_level: "expert"            # beginner, knowledgeable, expert
    years_experience: 10
    specific_insights:
      - "Training scaling laws hold consistently"
```

### Confidence Inference

The agent automatically infers confidence from your belief wording:

**High Confidence (0.80-0.95)**
- "will certainly", "inevitably", "must", "definitely"
- "historically always", "structurally guaranteed"

**Medium Confidence (0.60-0.75)**
- "likely", "probably", "expected to", "tends to"
- "typically", "in most cases", "generally"

**Low Confidence (0.35-0.50)**
- "might", "could", "possibly", "may"
- "uncertain", "unclear", "remains to be seen"

You can customize these phrases in the configuration.

## How It Works

### 1. Market Analysis Flow

```
Market arrives for analysis
    ↓
Check if worldview applies (category filter)
    ↓
Retrieve relevant passages from knowledge library (RAG)
    ↓
Fetch and interpret current news through worldview
    ↓
Generate prediction with inferred confidence
    ↓
Compare to ensemble consensus
    ↓
If divergence > threshold → Trigger manual review
    ↓
Present review dashboard with:
    - Ensemble reasoning
    - Worldview-based prediction
    - Retrieved knowledge
    - News interpretation
    - Your decision options
```

### 2. RAG Retrieval Process

The knowledge library:
1. **Chunks documents** into 512-token segments with 128-token overlap
2. **Generates embeddings** using OpenAI's text-embedding-3-small
3. **Stores in FAISS** index for fast similarity search
4. **Retrieves top-k passages** relevant to the market query
5. **Caches the index** to avoid re-processing on every run

### 3. News Interpretation

The agent reads news FROM your perspective:
- Takes in current events
- Interprets them THROUGH your ideological framework
- Identifies what confirms YOUR expectations
- Notes what would surprise YOU

It doesn't analyze your worldview - it uses it to understand the world.

## Usage Examples

### Example 1: AI Regulation Market

**Market**: "Will the US pass comprehensive AI regulation by 2026?"

**Ensemble Consensus**: 70% YES (based on current legislative momentum)

**Your Worldview Prediction**: 35% YES

**Retrieved Knowledge**:
- From `taleb_black_swan.pdf`: "Regulatory capture is structurally inevitable..."
- From your notes: "AI timeline predictions consistently underestimate bureaucratic friction"

**News Interpretation**:
- Headlines suggest momentum
- Your framework sees institutional incentives for delay
- Congressional calendar constraints often missed

**Override Trigger**: YES (25% divergence + high confidence)

### Example 2: Inflation Market

**Market**: "Will CPI exceed 4% in Q3 2025?"

**Ensemble Consensus**: 45% YES

**Your Worldview Prediction**: 25% YES

**Retrieved Knowledge**:
- Your monetarist framework + tech deflation notes

**Override Trigger**: MAYBE (20% divergence)

## Review Dashboard

When an override is triggered, you'll see:

```
┌─────────────────────────────────────────────────────────┐
│ DIVERGENCE DETECTED: AI Regulation Timeline             │
├─────────────────────────────────────────────────────────┤
│ ENSEMBLE CONSENSUS (70% confidence):                    │
│ "70% YES - Regulatory timeline likely extended due      │
│ to lobbying pressure and midterm politics"              │
│ Agents: Forecaster 0.72 | Bull 0.75 | Bear 0.65         │
├─────────────────────────────────────────────────────────┤
│ YOUR WORLDVIEW ANALYSIS:                                │
│ Confidence: 0.85 (high - "structurally inevitable")     │
│ Prediction: 35% YES                                     │
│                                                         │
│ Supporting theory:                                      │
│ 📚 "Regulatory capture is structurally inevitable..."   │
│    — Taleb, Antifragile (retrieved passage)             │
│                                                         │
│ Personal insight:                                       │
│ 📝 "AI timeline predictions underestimate friction"     │
│                                                         │
│ News interpretation:                                    │
│ 📰 Headlines suggest momentum, but framework sees       │
│    institutional incentives for delay                   │
├─────────────────────────────────────────────────────────┤
│ MARKET PRICE: 65¢ YES (suggests 65% probability)        │
├─────────────────────────────────────────────────────────┤
│ [TRADE WITH WORLDVIEW] [FOLLOW ENSEMBLE] [SKIP]         │
└─────────────────────────────────────────────────────────┘
```

## Best Practices

### 1. Be Honest About Expertise

Only claim expertise in domains where you genuinely have deep knowledge. The calibration tracking will expose overconfidence.

### 2. Update Your Worldview

Regularly review and update `worldview.yaml` based on outcomes:

```yaml
belief_updates:
  - date: "2025-01-15"
    market_id: "MARKET-XYZ"
    prior_belief: "AI capabilities were overestimated"
    outcome: "AI capabilities were underestimated"
    update_magnitude: 0.3
    new_belief: "Markets consistently underestimate AI progress"
```

### 3. Curate Your Knowledge Library

Quality matters more than quantity:
- Include foundational texts that shaped your thinking
- Add your own notes and frameworks
- Update with new evidence
- Remove sources you've changed your mind about

### 4. Start Conservative

Begin with:
- Lower confidence estimates
- Narrower domain expertise claims
- Higher override thresholds

Adjust as you build track record.

## Calibration Tracking

The system tracks your prediction accuracy:

```python
# View your calibration stats
from src.utils.database import DatabaseManager

db = DatabaseManager()
ideology_predictions = await db.get_ideology_predictions()

# Calculate Brier score
brier_score = calculate_brier_score(ideology_predictions)
print(f"Your Brier score: {brier_score:.3f} (lower is better)")
print(f"Ensemble Brier score: {ensemble_brier:.3f}")
```

Use this to:
- Identify domains where you add value
- Adjust confidence estimates
- Update your worldview

## Troubleshooting

### Knowledge Library Not Initializing

**Problem**: FAISS or OpenAI not available

**Solution**:
```bash
pip install faiss-cpu openai
# Or for GPU support:
pip install faiss-gpu openai
```

### No Passages Retrieved

**Problem**: Documents not indexed

**Solution**:
```python
from src.utils.knowledge_library import get_knowledge_library
kl = await get_knowledge_library()
await kl.refresh_index()
print(kl.get_library_stats())
```

### Override Triggers Too Often

**Solution**: Adjust thresholds in `worldview.yaml`:
```yaml
override_triggers:
  min_divergence_from_ensemble: 0.30  # Increase from 0.20
  min_worldview_confidence: 0.80      # Increase from 0.70
```

## Philosophy

The Ideology Agent embodies a key insight: **well-calibrated worldviews can provide genuine predictive edge**.

Unlike other agents that might question or critique, the Ideology Agent **fully inhabits your perspective**. It:
1. **Reasons FROM your worldview**, not ABOUT it
2. **Uses your knowledge library** as its source of understanding
3. **Generates predictions with conviction** from your framework
4. **Represents your view authentically** in the ensemble debate

The other agents (Forecaster, Bull, Bear, Risk Manager) provide the balancing perspectives. The Ideology Agent's job is to give YOUR view a voice - to represent how someone with your beliefs and expertise would genuinely see the market.

This creates a true ensemble where different viewpoints compete, and the Trader agent synthesizes them into a final decision.

## Integration with Other Agents

The Ideology Agent participates in the debate as ONE VOICE among many:

```
Step 0: Forecaster (objective probability estimate)
Step 0.5: News Analyst (sentiment analysis)
Step 0.5: Ideology Analyst (YOUR perspective) [NEW]
Step 1: Bull Researcher (optimistic case)
Step 2: Bear Researcher (pessimistic case)
Step 3: Risk Manager (risk assessment)
Step 4: Trader (synthesizes all views into final decision)
```

The Ideology Agent doesn't critique other views - it presents YOUR view. The Trader agent then weighs all perspectives (objective analysis, bull case, bear case, YOUR view, etc.) to reach a final decision.

## Future Enhancements

Potential future features:
- **Dynamic weighting**: Auto-adjust ideology weight based on calibration
- **Learning mode**: Identify which types of markets your worldview predicts well
- **Contrarian indicators**: Flag when your view aligns with market (potential overreaction)
- **Multi-worldview support**: Compare predictions across different frameworks

## Support

For issues or questions:
1. Check this documentation
2. Review logs in `logs/ideology_agent.log`
3. Examine retrieved passages and interpretations
4. Adjust configuration as needed

Remember: The goal is building a well-calibrated worldview, not proving you're right.