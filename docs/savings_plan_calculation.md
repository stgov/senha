# AWS Savings Plan - 1 Year Calculation

## Compute Savings Plan (1 Year - No Upfront Payment)

### Services Covered:
- **Amazon SageMaker**: ml.g4dn.xlarge endpoint (24/7, 720 hours/month)
- **AWS Lambda**: Compute usage (GB-seconds)

### Current On-Demand Costs (Base):
- SageMaker: $529.92 USD/month
- Lambda: $10.20 USD/month
- **Total Compute (On-Demand)**: $540.12 USD/month

### AWS Savings Plan Discounts (1 Year Term):
Based on AWS pricing documentation for Compute Savings Plans:

| Payment Option | Discount on SageMaker | Discount on Lambda |
|---------------|----------------------|-------------------|
| **No Upfront** | **17%** | **10%** |
| Partial Upfront | 20% | 12% |
| All Upfront | 24% | 15% |

### Estimated Costs with Savings Plan (No Upfront):

#### SageMaker ml.g4dn.xlarge:
- **On-Demand**: $529.92 USD/month
- **With 17% discount**: $529.92 × 0.83 = **$439.83 USD/month**
- **Monthly Savings**: $90.09 USD

#### AWS Lambda:
- **On-Demand**: $10.20 USD/month
- **With 10% discount**: $10.20 × 0.90 = **$9.18 USD/month**
- **Monthly Savings**: $1.02 USD

### Total Savings Calculation:
- **Total Compute Before**: $540.12 USD/month
- **Total Compute After**: $439.83 + $9.18 = **$449.01 USD/month**
- **Monthly Savings**: $91.11 USD/month
- **Annual Savings**: $1,093.32 USD/year

### Updated Monthly Costs:

**Before Savings Plan:**
```
Kinesis Video Streams:  $133.50
Lambda:                  $10.20
SageMaker:               $529.92
OpenSearch:              $4,343.75
S3:                      $11.51
SNS:                     $19.98
CloudWatch:              $120.23
───────────────────────────────
Total:                   $5,169.09 USD/month
```

**After Savings Plan (1 Year - No Upfront):**
```
Kinesis Video Streams:  $133.50
Lambda (with SP):        $9.18    (-$1.02)
SageMaker (with SP):     $439.83  (-$90.09)
OpenSearch:              $4,343.75
S3:                      $11.51
SNS:                     $19.98
CloudWatch:              $120.23
───────────────────────────────
Total:                   $5,077.98 USD/month
Monthly Savings:         $91.11 USD/month
```

### Annual Comparison:
- **12 Months Without Savings Plan**: $62,029.08 USD
- **12 Months With Savings Plan**: $60,935.76 USD
- **Total Annual Savings**: **$1,093.32 USD**

### Savings Plan Commitment:
- **Required Monthly Commitment**: ~$450 USD/month
- **Coverage**: All SageMaker and Lambda compute usage
- **Flexibility**: Can be applied across different instance families and regions

### How to Apply:
1. Go to **AWS Billing Console** → **Savings Plans**
2. Select **Compute Savings Plan**
3. Configure:
   - **Term**: 1 Year
   - **Payment Option**: No Upfront (Monthly payments)
   - **Commitment**: ~$450 USD/month (will automatically cover your usage)
4. The discount applies automatically to all eligible compute usage

### Notes:
- Savings Plan pricing replaces on-demand pricing for covered services
- Lambda discounts apply to compute (GB-seconds), not requests
- SageMaker endpoint hours receive the full 17% discount
- The commitment covers usage up to the committed amount, any overage is charged at on-demand rates

