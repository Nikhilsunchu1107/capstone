# Hand-check sheet (15 rows). For each row write: is the answer correct vs the gold? y/n

## 1. Vector RAG / q08 (A_single_fact)
**Q:** Under the Cyber Shield policy, for how many days can lost wages be claimed after a cyber bullying or defamation event?

**Gold:** For up to a maximum of 7 working days.

**Answer:** Under the Cyber Shield policy, lost wages can be claimed for up to a maximum of 7 working days after a cyber bullying or defamation event.

Judge: faithfulness=1.0, answer_relevancy=1.0, context_precision=0.2, context_recall=1.0, answer_correctness=1.0

**Your verdict (answer correct? y/n):** 

## 2. Vector RAG / q02 (A_single_fact)
**Q:** Under the Trade Credit Insurance policy, when can the Insured apply for the No Claims Bonus?

**Gold:** After six (6) months from the expiration of the insurance period.

**Answer:** Under the Trade Credit Insurance policy, the application for No Claims Bonus shall be submitted after six (6) months from the expiration of the insurance period.

Judge: faithfulness=1.0, answer_relevancy=1.0, context_precision=0.2, context_recall=1.0, answer_correctness=1.0

**Your verdict (answer correct? y/n):** 

## 3. Vector RAG / q18 (B_multi_part)
**Q:** Under the Arogya Sanjeevani policy, what are the claim notification deadlines for emergency cashless, planned cashless, and reimbursement claims?

**Gold:** Emergency cashless: within 24 hours of hospitalization or before discharge, whichever is earlier. Planned cashless: at least 72 hours before hospitalization. Reimbursement: within 48 hours of admission or before discharge, whichever is earlier.

**Answer:** Under the Arogya Sanjeevani policy, the claim notification deadlines are:

- **Emergency cashless hospitalization:** Within 24 hours from the date of emergency hospitalization required, or before the Insured Person's discharge from Hospital, whichever is earlier.
- **Planned cashless hospitalization:** At least 72 hours prior to the planned Hospitalization.
- **Reimbursement claims:** Within 48 hours of admission or before discharge from the Hospital, whichever is earlier.

Judge: faithfulness=1.0, answer_relevancy=1.0, context_precision=0.2, context_recall=1.0, answer_correctness=1.0

**Your verdict (answer correct? y/n):** 

## 4. Vector RAG / q16 (A_single_fact)
**Q:** Under the Tata AIG Travel Insurance International Plus Health policy, for what time unit is the Hijack/Kidnap Daily Allowance paid?

**Gold:** A fixed amount for every 12-hour period, up to the maximum number of days in the Policy Schedule.

**Answer:** Under the Tata AIG Travel Insurance International Plus Health policy, the Hijack/Kidnap Daily Allowance is paid for every 12 hours period.

Judge: faithfulness=1.0, answer_relevancy=1.0, context_precision=0.2, context_recall=1.0, answer_correctness=1.0

**Your verdict (answer correct? y/n):** 

## 5. Vector RAG / q15 (A_single_fact)
**Q:** What is the current compounding interest rate on a policy loan under the HDFC Life Smart Pension Plus policy?

**Gold:** 9.50% per annum.

**Answer:** The current compounding interest rate on a policy loan under the HDFC Life Smart Pension Plus policy is 9.50% p.a.

Judge: faithfulness=1.0, answer_relevancy=1.0, context_precision=0.2, context_recall=1.0, answer_correctness=1.0

**Your verdict (answer correct? y/n):** 

## 6. PageIndex RAG / q09 (A_single_fact)
**Q:** Under the Port Package policy, within how many days must the insured notify the insurer of newly acquired equipment that was not previously declared?

**Gold:** Within 28 days of acquisition.

**Answer:** Under the Port Package policy, the insured must notify the insurer within 28 days of acquisition.

Judge: faithfulness=1.0, answer_relevancy=1.0, context_precision=1.0, context_recall=1.0, answer_correctness=1.0

**Your verdict (answer correct? y/n):** 

## 7. PageIndex RAG / q07 (A_single_fact)
**Q:** Under the Click 2 Protect Optima Secure policy, what is payable if the life assured dies by suicide within 12 months of risk commencement?

**Gold:** At least 80% of the total premiums paid till the date of death, or the surrender value on the date of death, whichever is higher, provided the policy is in force.

**Answer:** Under the Click 2 Protect Optima Secure policy, if death is due to suicide within 12 months from the date of commencement of risk under the policy or from the date of revival of the policy, as applicable, the nominee or beneficiary of the Policyholder shall be entitled to at least 80% of the total Premiums paid till the date of death or the surrender value available as on the date of death, whichever is higher, provided the policy is in force.

Judge: faithfulness=1.0, answer_relevancy=1.0, context_precision=0.6666666666666666, context_recall=1.0, answer_correctness=1.0

**Your verdict (answer correct? y/n):** 

## 8. PageIndex RAG / q35 (D_cross_document)
**Q:** For a period not exceeding 8 months, which policy has the higher short-period rate: Cyber Shield or Specie Insurance?

**Gold:** Cyber Shield. Its rate is 85% of the annual rate. The Specie Insurance rate is 80% of the annual rate.

**Answer:** For a period not exceeding 8 months, the Specie Insurance policy has the higher short-period rate: 80% of the Annual rate, compared to 85% for Cyber Shield.

Judge: faithfulness=0.67, answer_relevancy=0.7, context_precision=0.8, context_recall=1.0, answer_correctness=0.5

**Your verdict (answer correct? y/n):** 

## 9. PageIndex RAG / q06 (A_single_fact)
**Q:** Under the Bharat Griha Raksha policy, within how many days must the insured submit the claim form?

**Gold:** Within 30 days from the date the insured first notices the loss or damage.

**Answer:** Under the Bharat Griha Raksha policy, the insured must submit the claim form at the earliest opportunity, but within 30 days from the date they first notice the loss or damage.

Judge: faithfulness=1.0, answer_relevancy=1.0, context_precision=0.333, context_recall=1.0, answer_correctness=1.0

**Your verdict (answer correct? y/n):** 

## 10. PageIndex RAG / q38 (D_cross_document)
**Q:** Compare the grace period in the Click 2 Protect Optima Secure policy and the Arogya Sanjeevani policy. Is cover active during the grace period in each?

**Gold:** Click 2 Protect: 30 days (15 days for monthly mode), and the policy is considered in force with the risk cover. Arogya Sanjeevani: 30 days for renewal, and coverage is not available during the grace period.

**Answer:** **Click 2 Protect Optima Secure (HDFC Life Click 2 Protect Life):**

- Grace period: **30 days** where the mode of payment of Premium is other than monthly, and **15 days** in case of monthly mode, allowed for the payment of each renewal Premium after the first Premium.
- Cover during grace period: The policy is considered to be **in-force with the risk cover during the grace period without any interruption**. Should a valid claim arise under the Policy during the grace period, but before the payment of due premium, the claim shall still be honored; in such cases, the due and unpaid premium for the policy year will be deducted from any benefit payable.

**Arogya Sanjeevani (SBI General Insurance):**

- Grace period: In case of monthly instalment option, a Grace Period of **15 days** is applicable. In case of Single, Half Yearly, Quarterly instalment options, a Grace Period of **30 days** is applicable. (Also, at the end of the Policy Period, the Policy can be renewed within the Grace Period of 30 days to maintain continuity of benefits without Break in Policy.)
- Cover during grace period: For premium payment in instalments, **coverage will be available from the due date of instalment premium till the date of receipt of premium by Company**. However, for renewal at the end of the Policy Period, **Coverage is not available during the Grace Period**.

**Comparison:**

- The grace period durations are the same in both policies: **15 days for monthly mode** and **30 days for other modes**.
- On whether cover is active during the grace period: Under Click 2 Protect Optima Secure, the policy is in-force with the risk cover during the grace period without any interruption. Under Arogya Sanjeevani, coverage is available during the grace period for instalment premiums (from the due date of instalment premium till the date of receipt of premium by the Company), but for renewal at the end of the Policy Period, coverage is **not** available during the grace period.

Judge: faithfulness=1.0, answer_relevancy=1.0, context_precision=1.0, context_recall=1.0, answer_correctness=1.0

**Your verdict (answer correct? y/n):** 

## 11. Graph RAG (linked) / q28 (B_multi_part)
**Q:** Under the Contractors Plant and Machinery Insurance policy, what must the insured do after an occurrence that can give rise to a claim?

**Gold:** Immediately notify the Company by telephone or telegram and in writing, with the nature and extent of the loss. Take all reasonable steps to minimise the loss. Preserve the damaged or defective parts for inspection. Furnish all information and documentary evidence the Company requires.

**Answer:** Under the Contractors Plant and Machinery Insurance policy, in the event of any occurrence which might give rise to a claim, the Insured shall:

a) Immediately notify the Company by telephone or telegram as well as in writing, giving an indication as to the nature and extent of loss or damage.

b) Take all reasonable steps within his power to minimise the extent of the loss or damage or liability.

c) Preserve the damaged or defective parts and make them available for inspection by an official or surveyor of the Company.

d) Furnish all such information and documentary evidence as the company may require.

Judge: faithfulness=1.0, answer_relevancy=1.0, context_precision=0.2, context_recall=1.0, answer_correctness=1.0

**Your verdict (answer correct? y/n):** 

## 12. Graph RAG (linked) / q03 (A_single_fact)
**Q:** How long is the free look period under the Arogya Sanjeevani policy, and from when does it start?

**Gold:** 30 days, beginning from the date of receipt of the policy document.

**Answer:** Under the Arogya Sanjeevani policy, the free look period is 30 days, beginning from the date of receipt of the Policy document, whether received electronically or otherwise.

Judge: faithfulness=1.0, answer_relevancy=1.0, context_precision=0.6, context_recall=1.0, answer_correctness=1.0

**Your verdict (answer correct? y/n):** 

## 13. Graph RAG (linked) / q02 (A_single_fact)
**Q:** Under the Trade Credit Insurance policy, when can the Insured apply for the No Claims Bonus?

**Gold:** After six (6) months from the expiration of the insurance period.

**Answer:** Under the Trade Credit Insurance policy, the application for No Claims Bonus shall be submitted after six (6) months from the expiration of the insurance period.

Judge: faithfulness=1.0, answer_relevancy=1.0, context_precision=0.2, context_recall=1.0, answer_correctness=1.0

**Your verdict (answer correct? y/n):** 

## 14. Graph RAG (linked) / q06 (A_single_fact)
**Q:** Under the Bharat Griha Raksha policy, within how many days must the insured submit the claim form?

**Gold:** Within 30 days from the date the insured first notices the loss or damage.

**Answer:** Under the Bharat Griha Raksha policy, the insured must submit the claim form within 30 days from the date they first notice the loss or damage.

Judge: faithfulness=1.0, answer_relevancy=1.0, context_precision=0.2, context_recall=1.0, answer_correctness=1.0

**Your verdict (answer correct? y/n):** 

## 15. Graph RAG (linked) / q14 (A_single_fact)
**Q:** In the Weather Insurance (Retail) policy, what solar irradiance threshold defines Sunshine Hours?

**Gold:** The period during which direct solar irradiance exceeds 120 watts per square meter.

**Answer:** According to the context, Sunshine Hours is defined as the period during which direct solar irradiance exceeds the threshold value of 120 watts per square meter.

Judge: faithfulness=1.0, answer_relevancy=1.0, context_precision=0.2, context_recall=1.0, answer_correctness=1.0

**Your verdict (answer correct? y/n):** 
