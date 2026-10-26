# Hand-check sheet (15 rows). For each row write: is the answer correct vs the gold? y/n

## 1. Vector RAG / q114 (H_paraphrased)
**Q:** A crane at our container terminal is insured for less than it is worth on the market. When does the insurer scale down a damage payout, and when does it not?

**Gold:** The reduction (Average) applies only if the insured value is less than 80% of the market value of the item. The payout is then reduced by the percentage by which the insured value is below the market value. It does not apply if the insured value is an agreed value, and it does not apply to freight charges and labour costs.

**Answer:** The insurer scales down a damage payout under the Average clause. According to the context, if the market value of an item of equipment at the time of an accident is greater than the insured value, the insurer will reduce reimbursement by the percentage by which the insured value of the item is less than the market value.

This scaling down applies only if the insured value is less than 80% of the market value of the item (6.1). It applies separately to each item of equipment (6.2).

It does **not** apply:
- to freight charges and labour costs (6.3)
- if the insured value is an agreed value (6.4)

Also, the clause does not require the insurer to pay more than the insured value for any item of equipment (6.5).

Judge: faithfulness=1.0, answer_relevancy=1.0, context_precision=0.6, context_recall=1.0, answer_correctness=1.0

**Your verdict (answer correct? y/n):** 

## 2. Vector RAG / q103 (H_paraphrased)
**Q:** Under the Bharat Griha Raksha policy, my flat is worth more than the amount I insured it for. Will the payout be cut in proportion?

**Gold:** No. Underinsurance does not apply to this policy. If the sum insured is less than the actual value at risk, the difference does not affect the amount paid.

**Answer:** Under the TATA AIG Bharat Griha Raksha Policy, underinsurance does not apply. Thus, if your Sum Insured calculated on the basis of the information that you provided is less than the actual value at risk, the difference will not affect the amount We pay.

Judge: faithfulness=1.0, answer_relevancy=1.0, context_precision=0.13333333333333333, context_recall=1.0, answer_correctness=1.0

**Your verdict (answer correct? y/n):** 

## 3. Vector RAG / q55 (E_multi_hop)
**Q:** Under the Cyber Shield policy with Family Cover, is the policyholder's 20-year-old child treated as an insured person (You) or as a Third Party?

**Gold:** As a Third Party. Family Members are the legal spouse and at most two eldest children who are both below the age of 18. A 20-year-old child is not a Family Member. Third Party means any person except You and Your Family Members.

**Answer:** Under the Cyber Shield policy, the policyholder's 20-year-old child would be treated as a **Third Party**, not as an insured person (You).

According to the context:

- **You/Your** means the Policyholder/Certificate holder who is a permanent resident of India. If Family Cover is purchased, then You/Your means the Policyholder/Certificate holder and the Policyholder's/Certificate holder's **Family Members**, who are permanent residents of India.

- **Family Members** means the Policyholder's immediate family which includes a legal spouse and a **maximum of two children who are eldest, however both being below the age of 18**.

- **Third Party** means any natural person and/or legal person and/or association of persons **except You and Your Family Members**.

Since the child is 20 years old, they are **not below the age of 18** and therefore do not fall within the definition of Family Members. As a result, they are not included in the definition of You/Your and would be treated as a Third Party.

Judge: faithfulness=1.0, answer_relevancy=1.0, context_precision=0.2, context_recall=0.5, answer_correctness=1.0

**Your verdict (answer correct? y/n):** 

## 4. Vector RAG / q51 (E_multi_hop)
**Q:** Under the Bharat Griha Raksha policy, a flood damages an insured home. Five days later, goods are stolen from inside the home because of the flood damage. Is the theft covered, and whom must the insured inform?

**Gold:** Yes. Theft within 7 days from the occurrence of, and proximately caused by, an Insured Event is covered. Articles outside the home are not covered. The insured must inform the police.

**Answer:** Under the Bharat Griha Raksha policy, theft within 7 (seven) days from the occurrence of and proximately caused by an Insured Event (such as flood) is covered. Since the theft occurred five days after the flood, it falls within this 7-day period.

The insured must inform the police. The policy states that if there is a theft within 7 (seven) days following an Insured Event, you must inform the police.

Judge: faithfulness=1.0, answer_relevancy=1.0, context_precision=0.333, context_recall=1.0, answer_correctness=0.75

**Your verdict (answer correct? y/n):** 

## 5. Vector RAG / q48 (E_multi_hop)
**Q:** In the Auto Secure Commercial Vehicle Package Policy, the IDV of a vehicle is Rs 10,00,000 and the cost to retrieve and repair it after an accident is Rs 8,00,000. How is the claim treated, and what is the most the Company will pay?

**Gold:** It is treated as a Constructive Total Loss, because the cost exceeds 75% of the IDV (Rs 7,50,000). The Company's liability will not exceed the IDV stated in the Schedule less the value of the wreck.

**Answer:** The context does not contain the answer to how this specific claim is treated or the exact amount the Company will pay for these figures. However, the context does state that the insured vehicle shall be treated as a CTL if the aggregate cost of retrieval and/or repair of the vehicle, subject to terms and conditions of the policy, exceeds 75% of the IDV of the vehicle.

Judge: faithfulness=0.5, answer_relevancy=0.4, context_precision=0.2, context_recall=1.0, answer_correctness=0.0

**Your verdict (answer correct? y/n):** 

## 6. PageIndex RAG / q117 (H_paraphrased)
**Q:** An excavator on a coastal building site is partly under sea water when the tide comes in. Is the damage paid?

**Gold:** No. Loss or damage due to total or partial immersion in tidal waters is excluded.

**Answer:** The context does not contain the answer to whether damage to an excavator partly under sea water when the tide comes in is paid.

Judge: faithfulness=0.0, answer_relevancy=0.0, context_precision=0.6, context_recall=1.0, answer_correctness=0.0

**Your verdict (answer correct? y/n):** 

## 7. PageIndex RAG / q113 (H_paraphrased)
**Q:** We keep gold bars and banknotes in a vault under a valuables policy. One of our own staff steals some of them. Does the policy pay?

**Gold:** No. Theft or dishonesty committed by, or in collusion with, any employee of the insured is excluded.

**Answer:** The policy does not pay. Under Section 2, Exclusions, B, this insurance does not cover "theft or dishonesty committed by or in collusion with any principal shareholder (beneficial or otherwise), partner, director, officer, or any employee of the insured or any person to whom the insured property is entrusted or loaned." Since the theft was committed by one of your own staff (an employee of the insured), it falls within this exclusion.

Judge: faithfulness=1.0, answer_relevancy=1.0, context_precision=0.67, context_recall=1.0, answer_correctness=1.0

**Your verdict (answer correct? y/n):** 

## 8. PageIndex RAG / q89 (G_unanswerable)
**Q:** Under the Contractors Plant and Machinery Insurance policy, how long is the free look period?

**Gold:** The Contractors Plant and Machinery Insurance wording does not state this.

**Answer:** The context does not contain information about a free look period.

Judge: faithfulness=None, answer_relevancy=None, context_precision=None, context_recall=None, answer_correctness=1.0

**Your verdict (answer correct? y/n):** 

## 9. PageIndex RAG / q111 (H_paraphrased)
**Q:** My cover pays out from rainfall measured at a named recording site. If that site stops reporting, which numbers does the insurer use?

**Gold:** Data from the back-up data source or weather station. If that is also not available, the insurer uses the fallback methodology, which replaces missing data with data from the same dates in recent historical years.

**Answer:** If the weather data is not available from the reference data source/weather station, then the back-up data source/weather stations data will be used. If weather data from back-up data source/weather stations are also not available, then the weather data generated through fallback methodology will be used.

Under the fallback methodology, if the Data Provider does not publish the Daily observed weather data in respect of any calendar day during the Policy Period, the missing data shall be replaced with an average value calculated as follows: any missing data shall be replaced with data measured on the corresponding dates of the most recent available mutually agreed number of historical years on an iterative basis. If more than one data point is missing for the current year, all the replacement data shall be taken from the same historical year(s), so that if in any historical year data is available for one date corresponding to a missing data point in the current year, but is not available for another missing data point, that historical year's data shall not be used as replacement data. Each of the last mutually agreed number of historical year's available replacement data shall be applied in turn in place of each of the missing data points, and the loss payable under the Policy shall equate to the average of the mutually agreed number of amounts (if any) thus indicated as payable.

Judge: faithfulness=1.0, answer_relevancy=1.0, context_precision=1.0, context_recall=1.0, answer_correctness=1.0

**Your verdict (answer correct? y/n):** 

## 10. PageIndex RAG / q95 (G_unanswerable)
**Q:** Under the SBI General Livestock policy, after how many months of continuous coverage does the moratorium period begin?

**Gold:** The SBI General Livestock wording does not state this.

**Answer:** The context does not contain the answer.

Judge: faithfulness=None, answer_relevancy=None, context_precision=None, context_recall=None, answer_correctness=1.0

**Your verdict (answer correct? y/n):** 

## 11. Graph RAG / q74 (F_cross_document)
**Q:** Compare the free look period under the HDFC Life Smart Pension Plus policy and the Tata AIG Travel Insurance International Plus Health policy. How long is it, and to which policies does it apply?

**Gold:** Both give 30 days from receipt of the policy. HDFC Life Smart Pension Plus: it does not apply where the policyholder must compulsorily purchase the annuity from HDFC Life with the proceeds of a pension plan. Tata AIG Travel: it applies only to policies with a Policy Period of one year.

**Answer:** Under the HDFC Life Smart Pension Plus policy, the free look period is 30 days from the date of receipt of the policy, whether received electronically or otherwise.

Under the Tata AIG Travel Insurance International Plus Health policy, the free look period is applicable for policies with a Policy Period of one (1) year, and the Insured Person is allowed a period of thirty Days from the date of receipt of the Policy, whether received electronically or otherwise.

So both are 30 days, but the Tata AIG free look period applies only to policies with a Policy Period of one (1) year.

Judge: faithfulness=1.0, answer_relevancy=1.0, context_precision=0.333, context_recall=0.67, answer_correctness=0.67

**Your verdict (answer correct? y/n):** 

## 12. Graph RAG / q104 (H_paraphrased)
**Q:** Under the Trade Credit Insurance policy, a buyer does not pay an invoice. A director of that buyer is a first cousin of a director of my company. Is the loss covered?

**Gold:** No, unless the Insurer has agreed otherwise in writing. A blood relationship between directors of the Insured and the Counterparty, up to direct cousins, is a Controlling influence. The Insurer has no liability for losses where a Controlling influence can arise between the Insured and the Counterparty.

**Answer:** Under the Trade Credit Insurance policy, the loss is **not covered**.

The context states that the insurance does not apply if blood relationship exists between directors or largest shareholders of the Insured and the Counterparty, where blood relationship means direct blood relationship up to direct cousins (their parents are siblings), including cousins by marriage (spouses of one of the parent are siblings).

Since a director of the buyer (Counterparty) is a first cousin of a director of your company (the Insured), this falls within the excluded blood relationship, so the loss is not covered.

Judge: faithfulness=1.0, answer_relevancy=1.0, context_precision=0.13333333333333333, context_recall=1.0, answer_correctness=0.33

**Your verdict (answer correct? y/n):** 

## 13. Graph RAG / q103 (H_paraphrased)
**Q:** Under the Bharat Griha Raksha policy, my flat is worth more than the amount I insured it for. Will the payout be cut in proportion?

**Gold:** No. Underinsurance does not apply to this policy. If the sum insured is less than the actual value at risk, the difference does not affect the amount paid.

**Answer:** Under the TATA AIG Bharat Griha Raksha Policy, underinsurance does not apply. Thus, if your Sum Insured calculated on the basis of the information that you provided is less than the actual value at risk, the difference will not affect the amount We pay.

Judge: faithfulness=1.0, answer_relevancy=1.0, context_precision=0.13333333333333333, context_recall=1.0, answer_correctness=1.0

**Your verdict (answer correct? y/n):** 

## 14. Graph RAG / q111 (H_paraphrased)
**Q:** My cover pays out from rainfall measured at a named recording site. If that site stops reporting, which numbers does the insurer use?

**Gold:** Data from the back-up data source or weather station. If that is also not available, the insurer uses the fallback methodology, which replaces missing data with data from the same dates in recent historical years.

**Answer:** The context does not contain the answer.

Judge: faithfulness=0.0, answer_relevancy=0.0, context_precision=0.13333333333333333, context_recall=0.6666666666666666, answer_correctness=0.0

**Your verdict (answer correct? y/n):** 

## 15. Graph RAG / q47 (E_multi_hop)
**Q:** In the Auto Secure Commercial Vehicle Package Policy, a vehicle is three and a half years old. What depreciation percentage is used to fix its IDV, and what depreciation percentage applies to its other replaced parts in a partial loss claim?

**Gold:** 40% for fixing the IDV, and 25% on other replaced parts in a partial loss claim. The two schedules are different.

**Answer:** For fixing the IDV, a vehicle exceeding 3 years but not exceeding 4 years old has a depreciation rate of **40%**.

For other replaced parts in a partial loss claim, a vehicle exceeding 3 years but not exceeding 4 years old has a depreciation rate of **25%**.

Judge: faithfulness=1.0, answer_relevancy=1.0, context_precision=0.2, context_recall=1.0, answer_correctness=0.67

**Your verdict (answer correct? y/n):** 
