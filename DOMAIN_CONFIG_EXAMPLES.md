# Making HEXEval Domain-Agnostic

## Example Domain Configurations

### Credit Risk / Loan Default (Current)
```yaml
domain:
  name: "Credit Risk / Loan Default"
  prediction_task: "loan default risk"
  decision_verb: "approve or reject"
  decision_noun: "loan application"
  stakeholder_context: "at a financial institution"
  end_user_context: "applying for a loan to improve my life"
  positive_outcome: "loan approval"
  negative_outcome: "loan rejection"
```

### Customer Churn Prediction
```yaml
domain:
  name: "Customer Churn Prediction"
  prediction_task: "customer churn risk"
  decision_verb: "retain or let go"
  decision_noun: "customer account"
  stakeholder_context: "at a subscription business"
  end_user_context: "using this service and considering alternatives"
  positive_outcome: "retention"
  negative_outcome: "churn/cancellation"
```

### Medical Diagnosis
```yaml
domain:
  name: "Medical Diagnosis"
  prediction_task: "disease diagnosis"
  decision_verb: "diagnose and treat"
  decision_noun: "patient case"
  stakeholder_context: "at a healthcare facility"
  end_user_context: "seeking medical care for my health concern"
  positive_outcome: "early detection and treatment"
  negative_outcome: "missed diagnosis"
```

### Fraud Detection
```yaml
domain:
  name: "Fraud Detection"
  prediction_task: "fraud risk"
  decision_verb: "flag or approve"
  decision_noun: "transaction"
  stakeholder_context: "at a financial services company" 
  end_user_context: "making a legitimate transaction"
  positive_outcome: "transaction approved"
  negative_outcome: "false fraud flag"
```

### Hiring / Resume Screening
```yaml
domain:
  name: "Hiring / Resume Screening"
  prediction_task: "candidate fit"
  decision_verb: "interview or reject"
  decision_noun: "job application"
  stakeholder_context: "at a company"
  end_user_context: "applying for a job to advance my career"
  positive_outcome: "interview invitation"
  negative_outcome: "application rejection"
```

## How It Works

### Old (Hardcoded)
```python
intro = f"You are {name}, a {role} at a financial institution..."
task = "You're reviewing loan applications..."
```

### New (Configurable)
```python
stakeholder_ctx = domain_config.get('stakeholder_context', '')
prediction_task = domain_config.get('prediction_task', 'prediction')

intro = f"You are {name}, a {role} {stakeholder_ctx}..."
task = f"You're reviewing AI predictions for {prediction_task}..."
```

## Benefits

1. **Reusable across domains** - Just change config, no code changes
2. **Consistent evaluation** - Same personas, adapted to new context
3. **Easy to extend** - Add new domains in minutes
4. **Persona portability** - "Conservative decision maker" works for any domain

## To Use a New Domain

1. Edit `hexeval/config/eval_config.yaml`
2. Update the `domain:` section
3. Run - prompts adapt automatically!

No code changes needed! 🎉
