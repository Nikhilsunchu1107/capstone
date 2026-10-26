# Eval set v2: 80 new, harder questions (DRAFT, for review)

Ids q41 to q120. Tiers: E multi-hop in one policy (24) · F cross-document (24: 16 with 2 PDFs, 8 with 3 PDFs) · G unanswerable / trap (12) · H paraphrased (20: 10 name the policy, 10 do not).

Every quote was checked by `python -m eval.verify_quotes v2`: it is on the stated page of the stated PDF. For tier G, the script checked that the key term is absent from the named PDF and present in the trap PDF.

Reply with the ids to reject or change. After your approval the file is frozen with a sha256 and no question is edited.

---

## Tier E · multi-hop, one policy

### q41 · E_multi_hop
**Q:** Under the Trade Credit Insurance policy, a Counterparty becomes insolvent. Within what period must the Insured submit a complete claim, and from which date does that period run?

**Gold:** Within 60 days after the Date of Loss. For a loss due to Insolvency, the Date of Loss is the Due Date, which is the payment date originally set by the Contract.

- `14..Trade credit insc_GEN756.pdf` p7: "the Insured must submit in writing a complete claim in the form required by the Insurer (including all the information requested therein) any time within 60 (sixty) days after the Date of Loss"
- `14..Trade credit insc_GEN756.pdf` p14: "in the event of a loss due to Political Risk, Insolvency or refusal of payment by the Counterparty to the Contract - the Due Date"
- `14..Trade credit insc_GEN756.pdf` p15: "means the date of payment of the Commercial credit originally set by Contract"

### q42 · E_multi_hop
**Q:** Under the Trade Credit Insurance policy, the Insured's payments record with a Counterparty for the recent one year is INR 1,000 million. What is the highest Preliminary limit the Insured may set, and for how long may it be applied if the Insurer has not yet determined a Credit limit?

**Gold:** INR 200 million. One quarter of the payments record is INR 250 million, but the Preliminary limit cannot exceed INR 200 million. It may be applied for 3 months from the date it was set up, unless the Insurer restricts it.

- `14..Trade credit insc_GEN756.pdf` p3: "The Insured may establish up to 1/4 of the aggregate payments record of the recent one (1) year with a Counterparty"
- `14..Trade credit insc_GEN756.pdf` p3: "the Preliminary limit of such Counterparty does not exceed INR 200 million"
- `14..Trade credit insc_GEN756.pdf` p4: "the Preliminary limit may be applied for 3 (three) months from the date of setting up the Preliminary limit"

### q43 · E_multi_hop
**Q:** Under the Trade Credit Insurance policy, after a claim payment the Insured recovers INR 100 from the Counterparty and spends INR 10 as reasonable collection expenses. The insurance indemnity was INR 60 and the amount of outstanding claims was INR 80. How much is owed to the Insurer, and within how many days must the recovery be disbursed?

**Gold:** INR 67.5. The formula is (Recoveries - Reasonable Expenses for Collection) x (Insurance Indemnity / Amount of Outstanding Claims) = (100 - 10) x (60 / 80). The recovery must be disbursed within 40 days from the day of recovery.

- `14..Trade credit insc_GEN756.pdf` p9: "The amount owed to the Insurer = (Recoveries - Reasonable Expenses for Collection) * (Insurance Indemnity/Amount of Outstanding Claims)"
- `14..Trade credit insc_GEN756.pdf` p9: "such Recoveries shall be disbursed between the Insurer and the Insured within 40 (Forty) days commencing from the day of recovery"

### q44 · E_multi_hop
**Q:** Under the Arogya Sanjeevani policy, the sum insured is Rs 1,00,000 and all waiting periods are complete. Cataract surgery on one eye costs Rs 30,000. How much does the Company pay?

**Gold:** Rs 23,750. The cataract limit is 25% of the sum insured or Rs 40,000, whichever is lower, per eye, so Rs 25,000 is admissible. Every claim then bears a 5% co-payment, which leaves Rs 23,750.

- `Arogya_Sanjeevani_Policy_Wording_KEN_034037d936.pdf` p4: "subject to a limit of 25% of Sum Insured or 40,000/-, whichever is lower, per each eye in one policy year"
- `Arogya_Sanjeevani_Policy_Wording_KEN_034037d936.pdf` p8: "Each and every claim under the Policy shall be subject to a Co-payment of 5% applicable to claim amount admissible and payable"

### q45 · E_multi_hop
**Q:** Under the Arogya Sanjeevani policy, the sum insured is Rs 2,00,000. The insured person takes a room that costs Rs 8,000 per day. What room rent is admissible per day, and how are the other hospital expenses paid?

**Gold:** Rs 4,000 per day is admissible (2% of the sum insured, which is below the Rs 5,000 per day cap). All other hospital expenses, except the cost of medicines, are paid in the same proportion as the admissible rate bears to the actual rate, which is 50% here.

- `Arogya_Sanjeevani_Policy_Wording_KEN_034037d936.pdf` p4: "Room Rent, Boarding, Nursing Expenses as provided by the Hospital / Nursing Home up to 2% of the sum insured subject to maximum"
- `Arogya_Sanjeevani_Policy_Wording_KEN_034037d936.pdf` p4: "shall be eﬀected in the same proportion as the admissible rate per day bears to the actual rate per day of Room Rent/ ICU/ICCU charges"

### q46 · E_multi_hop
**Q:** Under the Arogya Sanjeevani policy, a disease was diagnosed by a physician two years before the first policy started. Is it a Pre-Existing Disease, and when will its treatment be covered?

**Gold:** Yes. A condition diagnosed not more than 36 months before the policy commencement is a Pre-Existing Disease. Its treatment is excluded until 36 months of continuous coverage from the first policy. After that it is covered only if it was declared at the time of application and accepted by the insurer.

- `Arogya_Sanjeevani_Policy_Wording_KEN_034037d936.pdf` p3: "that is/are diagnosed by a physician not more than 36 months prior to the date of commencement of the policy issued by the insurer"
- `Arogya_Sanjeevani_Policy_Wording_KEN_034037d936.pdf` p5: "shall be excluded until the expiry of 36 months  of continuous coverage after the date of inception of the ﬁrst policy with us"
- `Arogya_Sanjeevani_Policy_Wording_KEN_034037d936.pdf` p5: "for any pre-existing disease is subject to the same being declared at the time of application and accepted by us"

### q47 · E_multi_hop
**Q:** In the Auto Secure Commercial Vehicle Package Policy, a vehicle is three and a half years old. What depreciation percentage is used to fix its IDV, and what depreciation percentage applies to its other replaced parts in a partial loss claim?

**Gold:** 40% for fixing the IDV, and 25% on other replaced parts in a partial loss claim. The two schedules are different.

- `Auto_Secure_Commercial_Vehicle_Package_Policy_Base_Policy_Wording_22b7905015.pdf` p3: "Exceeding 3 years but not exceeding 4 years 40%"
- `Auto_Secure_Commercial_Vehicle_Package_Policy_Base_Policy_Wording_22b7905015.pdf` p2: "Exceeding 3 years but not exceeding 4 years 25%"

### q48 · E_multi_hop
**Q:** In the Auto Secure Commercial Vehicle Package Policy, the IDV of a vehicle is Rs 10,00,000 and the cost to retrieve and repair it after an accident is Rs 8,00,000. How is the claim treated, and what is the most the Company will pay?

**Gold:** It is treated as a Constructive Total Loss, because the cost exceeds 75% of the IDV (Rs 7,50,000). The Company's liability will not exceed the IDV stated in the Schedule less the value of the wreck.

- `Auto_Secure_Commercial_Vehicle_Package_Policy_Base_Policy_Wording_22b7905015.pdf` p3: "The insured vehicle shall be treated as a CTL if the aggregate cost of retrieval and/or repair of the vehicle, subject to terms and conditions of the policy, exceeds 75% of the IDV of the vehicle"
- `Auto_Secure_Commercial_Vehicle_Package_Policy_Base_Policy_Wording_22b7905015.pdf` p7: "as specified in the Schedule less the value of the wreck"

### q49 · E_multi_hop
**Q:** In the Auto Secure Commercial Vehicle Package Policy, an insured sells the vehicle after three consecutive claim-free years and takes a fresh insurance two years later. Can a No Claim Bonus be granted, and at what rate?

**Gold:** Yes, 35%. The bonus may be granted on a fresh insurance effected within 3 years from the date of sale, at the rate earned at the expiry of the last 12 months of insurance. Three consecutive claim-free years earn 35%.

- `Auto_Secure_Commercial_Vehicle_Package_Policy_Base_Policy_Wording_22b7905015.pdf` p9: "No claim made or pending during the preceding 3 consecutive years of insurance 35%"
- `Auto_Secure_Commercial_Vehicle_Package_Policy_Base_Policy_Wording_22b7905015.pdf` p9: "provided such fresh insurance is effected within 3 (three) years from the date of sale of insured vehicle"

### q50 · E_multi_hop
**Q:** Under the Bharat Griha Raksha policy, a home is insured for Rs 60 lakh under the Home Building Cover, and the Home Contents Cover is also opted. How much built-in cover applies to General Contents, and does it include jewellery?

**Gold:** Rs 10 lakh. The built-in cover is 20% of the Home Building sum insured (Rs 12 lakh here) but it is capped at Rs 10 lakh. It does not include jewellery. Jewellery is a Valuable Content, which is not covered unless the Optional Cover for Valuable Contents is purchased.

- `Bharat_Griha_Raksha_Policy_Policy_Wordings_5219f40e18.pdf` p9: "equal to 20% of the Sum Insured for Home Building Cover subject to a maximum of ₹ 10 Lakh (Rupees Ten Lakh)"
- `Bharat_Griha_Raksha_Policy_Policy_Wordings_5219f40e18.pdf` p8: "Valuable Contents of your Home are not covered under this Policy unless you have purchased the Optional Cover for the Valuable Contents"
- `Bharat_Griha_Raksha_Policy_Policy_Wordings_5219f40e18.pdf` p3: "Valuable Contents of your Home consist of items such as jewellery"

### q51 · E_multi_hop
**Q:** Under the Bharat Griha Raksha policy, a flood damages an insured home. Five days later, goods are stolen from inside the home because of the flood damage. Is the theft covered, and whom must the insured inform?

**Gold:** Yes. Theft within 7 days from the occurrence of, and proximately caused by, an Insured Event is covered. Articles outside the home are not covered. The insured must inform the police.

- `Bharat_Griha_Raksha_Policy_Policy_Wordings_5219f40e18.pdf` p6: "Theft within 7 (seven) days from the occurrence of and proximately caused by any of the above Insured Events"
- `Bharat_Griha_Raksha_Policy_Policy_Wordings_5219f40e18.pdf` p14: "If there is a theft within 7 (seven) days following an Insured Event you must inform the police"

### q52 · E_multi_hop
**Q:** Under the Click 2 Protect Optima Secure policy with the Life & CI Rebalance option, the Basic Sum Assured is Rs 1 crore and the Policy Term is 20 years. What are the Life Cover and Critical Illness sums assured at inception, and by how much do they change at each policy anniversary?

**Gold:** At inception the Life Cover SA is Rs 80 lakh (80%) and the CI SA is Rs 20 lakh (20%). At each policy anniversary the CI SA rises and the Life Cover SA falls by Rs 1.5 lakh, which is 30% x Basic Sum Assured / Policy Term.

- `Click-2-Protect-Optima-Secure-Policy-Bond-101Y122V05.pdf` p9: "Life Cover SA = 80% of Basic Sum Assured and CI SA = 20% of Basic Sum Assured"
- `Click-2-Protect-Optima-Secure-Policy-Bond-101Y122V05.pdf` p9: "This amount will be calculated as: 30% ×Basic Sum Assured ÷Policy Term"

### q53 · E_multi_hop
**Q:** Under the Click 2 Protect Optima Secure policy, a policy lapses. Within what time can it be revived and at what interest rate? If the life assured then dies by suicide six months after the revival, what is payable?

**Gold:** The revival application must be made within five years from the due date of the first unpaid premium and before the Policy Term ends, with outstanding premiums and interest, currently 9.5% p.a. compounded annually. The suicide exclusion also runs for 12 months from the date of revival, so the nominee gets at least 80% of the total premiums paid till death or the surrender value on the date of death, whichever is higher, provided the policy is in force.

- `Click-2-Protect-Optima-Secure-Policy-Bond-101Y122V05.pdf` p15: "the application for the revival should be made within five years from the due date of the first unpaid Premium and before the expiry of the Policy Term"
- `Click-2-Protect-Optima-Secure-Policy-Bond-101Y122V05.pdf` p15: "The current rate of interest is 9.5% p.a. compounded annually"
- `Click-2-Protect-Optima-Secure-Policy-Bond-101Y122V05.pdf` p17: "In case of death due to suicide within 12 months from the date of commencement of risk under the policy or from the date of revival of the policy"

### q54 · E_multi_hop
**Q:** Under the Cyber Shield policy, an employed insured has a daily salary rate of INR 8,000 and misses 10 working days because of a cyber bullying event. What is the highest amount of lost wages payable?

**Gold:** INR 45,500. Lost wages are paid for at most 7 working days, and the daily rate cannot be more than INR 6,500 per day (7 x 6,500).

- `Cyber_Shield_Policy_Wordings_78baa23b5a.pdf` p2: "Lost wages for up to a maximum of 7 working days"
- `Cyber_Shield_Policy_Wordings_78baa23b5a.pdf` p4: "Lost wages will be calculated based on the daily rate of Your last drawn monthly salary, but not more than INR 6,500 per day"

### q55 · E_multi_hop
**Q:** Under the Cyber Shield policy with Family Cover, is the policyholder's 20-year-old child treated as an insured person (You) or as a Third Party?

**Gold:** As a Third Party. Family Members are the legal spouse and at most two eldest children who are both below the age of 18. A 20-year-old child is not a Family Member. Third Party means any person except You and Your Family Members.

- `Cyber_Shield_Policy_Wordings_78baa23b5a.pdf` p4: "which includes a legal spouse and a maximum of two children who are eldest, however both being below the age of 18"
- `Cyber_Shield_Policy_Wordings_78baa23b5a.pdf` p5: "Third Party means any natural person and/ or legal person and/or association of persons except You and Your Family Members"

### q56 · E_multi_hop
**Q:** Under the Political Risk Insurance for Investors policy, a covered loss under Insuring Agreement III is 1,000,000, the Deductible is 100,000 and the Insured Percentage is 90%. How much does the Company indemnify, before policy limits?

**Gold:** 810,000. The Deductible is subtracted first, and the Insured Percentage is then applied to the remainder: (1,000,000 - 100,000) x 90%.

- `Policy_Wordings_political_risk_insurance_for_investors.pdf_b7a0e7c805.pdf` p5: "The Insured Percentage shall be applied to the amount of all covered losses after subtracting any applicable Deductible"
- `Policy_Wordings_political_risk_insurance_for_investors.pdf_b7a0e7c805.pdf` p3: "To indemnify the Insured for the Insured Percentage of the Insured's Loss in excess of the Deductible"

### q57 · E_multi_hop
**Q:** Under the Political Risk Insurance for Investors policy, by when must the Insured submit a written proof of Loss, and by when must any legal action against the Company be commenced?

**Gold:** The proof of Loss must be submitted within twelve months after the Date of Loss, or within 30 days after a demand by the Company. Any action against the Company must be commenced within twenty-four months following the Date of Loss.

- `Policy_Wordings_political_risk_insurance_for_investors.pdf_b7a0e7c805.pdf` p14: "twelve (12) months after the Date of Loss"
- `Policy_Wordings_political_risk_insurance_for_investors.pdf_b7a0e7c805.pdf` p14: "the Insured shall file such proof of Loss with the Company within 30 days thereafter"
- `Policy_Wordings_political_risk_insurance_for_investors.pdf_b7a0e7c805.pdf` p17: "unless such action is commenced within twenty-four (24) months following the Date of Loss"

### q58 · E_multi_hop
**Q:** Under the SBI General Livestock policy, an insured animal dies while it is transported 120 km by road. Is the death covered? Would the answer change if the animal was transported by air?

**Gold:** By road it is covered only if Add-On cover 4.3, Transit Cover Beyond 80 Kms, is opted. Transportation beyond 80 km is otherwise excluded. Transportation by air or sea is excluded in every case.

- `SBI_General_Livestock_Policy_Wording_38ef0b201b.pdf` p3: "Transportation beyond 80kms unless Add-On cover 4.3, Transit Cover Beyond 80kms, is opted under the Policy"
- `SBI_General_Livestock_Policy_Wording_38ef0b201b.pdf` p3: "5. Transportation by Air/Sea."

### q59 · E_multi_hop
**Q:** Under the SBI General Livestock policy, an insured animal dies. What must the Insured do about the carcass and about information for the Company, and how long does the Company have to reject a claim?

**Gold:** The Insured must give immediate notice and give the Company an opportunity to inspect the carcass until at least 24 hours after the notice. Within 7 days the Insured must furnish the information the Company requires. If the Company rejects the claim, it must do so within 30 days from receipt of all necessary documents.

- `SBI_General_Livestock_Policy_Wording_38ef0b201b.pdf` p4: "shall give the Company an opportunity of inspecting the carcass until at least the expiration of 24 hours after such notice shall have been given to the Company"
- `SBI_General_Livestock_Policy_Wording_38ef0b201b.pdf` p4: "The Insured shall also within 7 days furnish to the Company such information"
- `SBI_General_Livestock_Policy_Wording_38ef0b201b.pdf` p5: "the Company shall do so within a period of 30 days from the date of receipt of all necessary documents"

### q60 · E_multi_hop
**Q:** Under the HDFC Life Smart Pension Plus policy, the available surrender value is Rs 5,00,000, the annual annuity is Rs 60,000 and the loan interest rate is 9.50% p.a. What is the largest loan that can be granted?

**Gold:** About Rs 3.16 lakh (Rs 3,15,789). Two limits apply and the lower one governs. The loan cannot exceed 80% of the surrender value (Rs 4,00,000). The annual interest on the loan also cannot exceed 50% of the annual annuity (Rs 30,000), which at 9.50% allows a loan of about Rs 3,15,789.

- `hdfc-life-smart-pension-plus-v13-policy-document-individual.pdf` p22: "shall not exceed 80% of the available surrender value at that point of time"
- `hdfc-life-smart-pension-plus-v13-policy-document-individual.pdf` p22: "the effective annual interest amount payable on loan does not exceed 50% of the annual annuity amount payable under the policy"
- `hdfc-life-smart-pension-plus-v13-policy-document-individual.pdf` p22: "The current compounding interest rate on loan is 9.50% p.a"

### q61 · E_multi_hop
**Q:** Under the HDFC Life Smart Pension Plus policy, a Limited Pay policy with a 10-year deferment period is surrendered in policy year 3, after total premiums of Rs 6,00,000 were paid. What is the Guaranteed Surrender Value?

**Gold:** Rs 2,10,000. The GSV factor for policy year 3 is 35%, and GSV = GSV Factor x Total Premiums Paid. A Limited Pay policy acquires a GSV after at least two years' premiums, and GSV applies only where the deferment option is selected and the surrender is during the deferment period.

- `hdfc-life-smart-pension-plus-v13-policy-document-individual.pdf` p20: "on payment of at least two years’ premiums in case of Limited Pay"
- `hdfc-life-smart-pension-plus-v13-policy-document-individual.pdf` p20: "GSV = GSV Factor * Total Premiums Paid"
- `hdfc-life-smart-pension-plus-v13-policy-document-individual.pdf` p21: "3 35% 35%  35% 35% 35% 35% 35% 35% 35% 35% 35%"

### q62 · E_multi_hop
**Q:** Under the Tata AIG Travel Insurance International Plus Health policy, an Eligible Card is stolen. The insured reports it to the issuer 10 hours after discovering the theft. Some unauthorised charges were made 20 hours before the report. Are those charges reimbursed?

**Gold:** No. The report was in time, because it was within 12 hours of discovery. But the policy reimburses only unauthorised charges made up to 12 hours before the first report to the issuer. Charges made more than 12 hours before the report are excluded.

- `tata_aig_travel_insurance_international_plus_health_policy_wordings_012c2ec139.pdf` p20: "up to twelve (12) hours prior to Your ﬁrst reporting of the event to the issuer"
- `tata_aig_travel_insurance_international_plus_health_policy_wordings_012c2ec139.pdf` p20: "You must report the loss or theft of Your Eligible Card to the issuer(s) within twelve (12) hours after discovering Your Lost or Stolen Eligible Card event"
- `tata_aig_travel_insurance_international_plus_health_policy_wordings_012c2ec139.pdf` p20: "Unauthorised Charges made on the Eligible Card that was Lost or Stolen, more than twelve (12) hours prior to Your ﬁrst reporting the event to the issuer"

### q63 · E_multi_hop
**Q:** Under the Contractors Plant and Machinery Insurance policy, the cost to replace a machine with a new one is Rs 50 lakh, but the Sum Insured is Rs 40 lakh. A covered repair costs Rs 10 lakh. How much does the Company pay, before the Excess?

**Gold:** Rs 8 lakh. The Sum Insured must equal the new replacement cost. If it is less, the Company pays only in the proportion the sum insured bears to the amount required to be insured, which is 40/50 = 80% of Rs 10 lakh.

- `Policy_Wordings_contractors_plant_and_machinery_insurance.pdf_0175bcd067.pdf` p2: "the Sum Insured shall be equal to the cost of replacement of the insured property by new property of the same kind and same capacity"
- `Policy_Wordings_contractors_plant_and_machinery_insurance.pdf_0175bcd067.pdf` p3: "the Company will pay only in such proportion as the sum insured bears to the amount required to be insured"

### q64 · E_multi_hop
**Q:** Under the Weather Insurance (Retail) policy, must the insured file a claim form when the weather index deviates? By when is the payment made, and what data is used if the reference weather station has no data?

**Gold:** No claim form is required. The insurer calculates the loss from certified data. Payment is not earlier than the end of the risk period and not later than 30 days from the day the certified weather data is received. If the reference station has no data, the back-up data source is used. If that is also not available, the fallback methodology is used.

- `Weather_Insurance_Policy_Wordings_Retail_6a1bd806a7.pdf` p6: "shall not require the claim form and payment shall be made to the insured"
- `Weather_Insurance_Policy_Wordings_Retail_6a1bd806a7.pdf` p6: "not later than 30 days from the day certiﬁed weather data required for claim settlement is received"
- `Weather_Insurance_Policy_Wordings_Retail_6a1bd806a7.pdf` p7: "then the back-up data source/ weather stations data will be used"

---

## Tier F · cross-document

### q65 · F_cross_document
**Q:** Up to what repair cost can the insured authorise repairs without prior inspection under the Auto Secure Commercial Vehicle Package Policy, and under the Contractors Plant and Machinery Insurance policy?

**Gold:** Auto Secure: the estimated cost of repair, including replacements, must not exceed Rs 500. Contractors Plant and Machinery: minor damage not exceeding Rs 7,500, after the claim is notified.

- `Auto_Secure_Commercial_Vehicle_Package_Policy_Base_Policy_Wording_22b7905015.pdf` p3: "the estimated cost of such repair including replacements, does not exceed Rs. 500/-"
- `Policy_Wordings_contractors_plant_and_machinery_insurance.pdf_0175bcd067.pdf` p5: "the Insured may proceed with the repair of any minor damage not exceeding 7,500/-"

### q66 · F_cross_document
**Q:** For a repairable (partial) loss, is depreciation deducted on replaced parts under the Auto Secure Commercial Vehicle Package Policy and under the Contractors Plant and Machinery Insurance policy?

**Gold:** Auto Secure: yes, depreciation is deducted on replaced parts at fixed rates. Contractors Plant and Machinery: no deduction is made for depreciation on replaced parts, except parts with limited life.

- `Auto_Secure_Commercial_Vehicle_Package_Policy_Base_Policy_Wording_22b7905015.pdf` p2: "Subject to a deduction for depreciation at the rates mentioned below in respect of the parts replaced"
- `Policy_Wordings_contractors_plant_and_machinery_insurance.pdf_0175bcd067.pdf` p3: "No deduction shall be made for depreciation in respect of parts replaced, except those with limited life"

### q67 · F_cross_document
**Q:** After a claim is rejected, within what time must court action start before all benefits are forfeited under the Contractors Plant and Machinery Insurance policy, and under the Specie Insurance policy?

**Gold:** Contractors Plant and Machinery: within three months after the rejection. Specie Insurance: within twelve months after the rejection.

- `Policy_Wordings_contractors_plant_and_machinery_insurance.pdf_0175bcd067.pdf` p4: "if a claim is made and rejected and no action or suit is commenced within three months after such rejection"
- `specie_insurance_policy_wordings_8d69ba9234.pdf` p6: "if a claim is made and rejected and no court action or suit is commenced within twelve months after such rejection"

### q68 · F_cross_document
**Q:** If another insurance covers the same loss, how much does the insurer pay under the Contractors Plant and Machinery Insurance policy, and under the Political Risk Insurance for Investors policy?

**Gold:** Contractors Plant and Machinery: the Company pays no more than its rateable proportion of the loss. Political Risk: the policy is excess over any other valid bond, indemnity or insurance, so it pays only above the other insurance.

- `Policy_Wordings_contractors_plant_and_machinery_insurance.pdf_0175bcd067.pdf` p5: "the Company shall not be liable to pay or contribute more then its rateable proportion of such loss damage or liability"
- `Policy_Wordings_political_risk_insurance_for_investors.pdf_b7a0e7c805.pdf` p17: "The insurance provided under this policy shall be excess over any other valid bond, or other indemnity or insurance"

### q69 · F_cross_document
**Q:** Which disputes can go to arbitration under the Cyber Shield policy, and under the Political Risk Insurance for Investors policy?

**Gold:** Cyber Shield: only disputes about the quantum to be paid, when liability is otherwise admitted. No dispute is referable if the Company has disputed or not accepted liability. Political Risk: any and all disputes or differences arising under or in relation to the policy, including its existence, validity or termination.

- `Cyber_Shield_Policy_Wordings_78baa23b5a.pdf` p7: "If any dispute or diﬀerence shall arise as to the quantum to be paid under the policy (liability being otherwise admitted)"
- `Policy_Wordings_political_risk_insurance_for_investors.pdf_b7a0e7c805.pdf` p15: "Any and all disputes or differences which may arise under, out of, in connection with or in relation to this policy"

### q70 · F_cross_document
**Q:** Where must the insured event take place for medical treatment under the Arogya Sanjeevani policy, and for the Accidental Death benefit under the Tata AIG Travel Insurance International Plus Health policy?

**Gold:** Arogya Sanjeevani: all medical treatment must be taken in India only. Tata AIG Travel: the accident must occur whilst on the Insured Journey overseas.

- `Arogya_Sanjeevani_Policy_Wording_KEN_034037d936.pdf` p8: "All medical treatment for the purpose of this insurance will have to be taken in India only"
- `tata_aig_travel_insurance_international_plus_health_policy_wordings_012c2ec139.pdf` p12: "If You suﬀer an Injury solely and directly due to an Accident which occurs whilst on the Insured Journey overseas"

### q71 · F_cross_document
**Q:** How does a hijacking affect cover under the Tata AIG Travel Insurance International Plus Health policy, and under the Aviation Insurance policy?

**Gold:** Tata AIG Travel: it pays a fixed amount for every 12-hour period when the Common Carrier in which the insured travels is hijacked. Aviation Insurance: it does not apply to claims caused by hi-jacking or unlawful seizure of the aircraft.

- `tata_aig_travel_insurance_international_plus_health_policy_wordings_012c2ec139.pdf` p22: "if You are Kidnapped or any Common Carrier in which You are travelling has been Hijacked during the Insured journey/Trip"
- `Policy_Wordings_aviation_insurance.pdf_7b7e60200a.pdf` p12: "Hi-jacking or any unlawful seizure or wrongful exercise of control of the aircraft or crew in flight"

### q72 · F_cross_document
**Q:** How fast must a theft be reported to the police under the Auto Secure Commercial Vehicle Package Policy (vehicle theft), and under the Tata AIG Travel Insurance International Plus Health policy (lost or stolen Eligible Card)?

**Gold:** Auto Secure: report to the police and the insurer within 48 hours from the theft. Tata AIG Travel: file an FIR within 12 hours of discovering the loss.

- `Auto_Secure_Commercial_Vehicle_Package_Policy_Base_Policy_Wording_22b7905015.pdf` p10: "insured should report to the Police as well as insurer within 48 hrs from theft"
- `tata_aig_travel_insurance_international_plus_health_policy_wordings_012c2ec139.pdf` p21: "File an FIR within twelve (12) hours of discovering Your Loss"

### q73 · F_cross_document
**Q:** Compare the revival terms of a lapsed policy under the Click 2 Protect Optima Secure policy and the HDFC Life Smart Pension Plus policy: the time limit and the current interest rate.

**Gold:** Both are the same. Revival must be within five years from the due date of the first unpaid premium, and the current revival interest rate is 9.5% p.a. in both policies.

- `Click-2-Protect-Optima-Secure-Policy-Bond-101Y122V05.pdf` p15: "the application for the revival should be made within five years from the due date of the first unpaid Premium"
- `Click-2-Protect-Optima-Secure-Policy-Bond-101Y122V05.pdf` p15: "The current rate of interest is 9.5% p.a. compounded annually"
- `hdfc-life-smart-pension-plus-v13-policy-document-individual.pdf` p18: "within five years from the due date for payment of the first unpaid Premium"
- `hdfc-life-smart-pension-plus-v13-policy-document-individual.pdf` p19: "The current compounding rate of interest on revival is 9.5% p.a."

### q74 · F_cross_document
**Q:** Compare the free look period under the HDFC Life Smart Pension Plus policy and the Tata AIG Travel Insurance International Plus Health policy. How long is it, and to which policies does it apply?

**Gold:** Both give 30 days from receipt of the policy. HDFC Life Smart Pension Plus: it does not apply where the policyholder must compulsorily purchase the annuity from HDFC Life with the proceeds of a pension plan. Tata AIG Travel: it applies only to policies with a Policy Period of one year.

- `hdfc-life-smart-pension-plus-v13-policy-document-individual.pdf` p18: "within 30 days from the date of receipt of the policy whether received electronically or otherwise"
- `hdfc-life-smart-pension-plus-v13-policy-document-individual.pdf` p18: "Free look cancellation shall not be applicable where the Policyholder has to compulsorily purchase annuity from HDFC Life Insurance"
- `tata_aig_travel_insurance_international_plus_health_policy_wordings_012c2ec139.pdf` p45: "The Free Look Period will be applicable for policies with the Policy Period of one (1) year"

### q75 · F_cross_document
**Q:** How fast does the insurer respond to a grievance under the Click 2 Protect Optima Secure policy, and under the SBI General Livestock policy?

**Gold:** Click 2 Protect: within the regulatory turn around time of 14 days. SBI General Livestock: the Company aims to respond within 7 days, and tries to keep the regulatory TAT of 14 days.

- `Click-2-Protect-Optima-Secure-Policy-Bond-101Y122V05.pdf` p20: "will be responded to within the prescribed regulatory Turn Around Time (TAT) of 14 days"
- `SBI_General_Livestock_Policy_Wording_38ef0b201b.pdf` p5: "We aim to respond to all Grievances within 7 days"

### q76 · F_cross_document
**Q:** Which forum decides disputes under the Trade Credit Insurance policy, and under the Port Package policy?

**Gold:** Trade Credit: disputes are first resolved by negotiation. If that fails, the Delhi Arbitration Court decides them under Indian law. Port Package: the parties submit to the exclusive jurisdiction of the High Court of Mumbai.

- `14..Trade credit insc_GEN756.pdf` p13: "such disputes shall be resolved in accordance with the current legislation of  the Republic of India, in Delhi Arbitration Court"
- `PORT_PACKAGE_Policy_wording_a5f317091b.pdf` p162: "Each party agrees to submit to the exclusive jurisdiction of the High Court of Mumbai"

### q77 · F_cross_document
**Q:** Compare the initial waiting period for illness under the Arogya Sanjeevani policy with the initial waiting period for disease under the SBI General Livestock policy.

**Gold:** Arogya Sanjeevani: treatment of any illness within 30 days from the first policy commencement date is excluded, except claims arising from an accident. SBI General Livestock: death due to diseases contracted within 15 days from the Date of Commencement of Risk is excluded.

- `Arogya_Sanjeevani_Policy_Wording_KEN_034037d936.pdf` p5: "Expenses related to the treatment of any illness within 30 days from the ﬁrst policy commencement date shall be excluded"
- `SBI_General_Livestock_Policy_Wording_38ef0b201b.pdf` p2: "Death of insured Animal due to diseases contracted within 15 days from the Date of Commencement of Risk"

### q78 · F_cross_document
**Q:** Which exchange rate is used to convert a foreign-currency loss into Indian Rupees under the Cyber Shield policy, and under the Trade Credit Insurance policy?

**Gold:** Cyber Shield: the cash rate of exchange published on the RBI website on the date of the loss. Trade Credit: the Reserve Bank of India rate on the day the Insured Event occurred, but not higher than the rate on the day of delivery (or the invoice date for services).

- `Cyber_Shield_Policy_Wordings_78baa23b5a.pdf` p6: "the amount will be converted into INR at cash rate of exchange published on the website of the RBI"
- `14..Trade credit insc_GEN756.pdf` p14: "The rate of exchange used by the Reserve Bank of India on the day the Insured Event occurred will be used in calculating the amount of indemnification, subject to this not being higher than that of the day when delivery was effected"

### q79 · F_cross_document
**Q:** What happens to the insurance when the insured dies under the Bharat Griha Raksha policy, and under the Auto Secure Commercial Vehicle Package Policy?

**Gold:** Bharat Griha Raksha: the Home Building and Home Contents covers continue for the benefit of the legal representatives during the Policy Period. Auto Secure: the policy stays valid for three months from the death of the sole insured or until the policy expires, whichever is earlier. The legal heirs can apply in that period for a transfer or a new policy.

- `Bharat_Griha_Raksha_Policy_Policy_Wordings_5219f40e18.pdf` p13: "continue for the beneﬁt of your legal representative(s) during the Policy Period"
- `Auto_Secure_Commercial_Vehicle_Package_Policy_Base_Policy_Wording_22b7905015.pdf` p10: "will remain valid for a period of three months from the date of the death of insured or until the expiry of this policy (whichever is earlier)"

### q80 · F_cross_document
**Q:** How much notice must the insurer give to cancel the policy under the Political Risk Insurance for Investors policy, and under the Contractors Plant and Machinery Insurance policy?

**Gold:** Political Risk: 30 days written notice. Contractors Plant and Machinery: 15 days notice. In both cases the insurer returns a pro-rata or rateable part of the premium for the unexpired period.

- `Policy_Wordings_political_risk_insurance_for_investors.pdf_b7a0e7c805.pdf` p15: "The Company may cancel this policy by giving 30 days written notice of such cancellation to the last known address of the Insured"
- `Policy_Wordings_contractors_plant_and_machinery_insurance.pdf_0175bcd067.pdf` p6: "by 15 days notice to that eﬀect being given to the insured"

### q81 · F_cross_document
**Q:** When is a damaged item settled as a total loss instead of being repaired under the Auto Secure Commercial Vehicle Package Policy, the Port Package policy (handling equipment), and the Contractors Plant and Machinery Insurance policy?

**Gold:** Auto Secure: when the cost of retrieval and/or repair exceeds 75% of the IDV. Port Package: when the reasonable repair costs from one accident are estimated to exceed the insured value. Contractors Plant and Machinery: when the repair cost equals or exceeds the actual value of the machinery immediately before the damage.

- `Auto_Secure_Commercial_Vehicle_Package_Policy_Base_Policy_Wording_22b7905015.pdf` p3: "exceeds 75% of the IDV of the vehicle"
- `PORT_PACKAGE_Policy_wording_a5f317091b.pdf` p39: "If the reasonable costs of repair of damage sustained in one accident are estimated to exceed the insured value, we will pay the insured value as a CTL, less any salvage value"
- `Policy_Wordings_contractors_plant_and_machinery_insurance.pdf_0175bcd067.pdf` p3: "If the cost of repairs as detailed hereinabove equals or exceeds the actual value of the machinery insured immediately before the occurrence of the damage"

### q82 · F_cross_document
**Q:** What reward does the insured get for a claim-free period under the Arogya Sanjeevani policy, the Auto Secure Commercial Vehicle Package Policy, and the Trade Credit Insurance policy?

**Gold:** Arogya Sanjeevani: the Cumulative Bonus increases the sum insured by 5% for each claim-free policy year. Auto Secure: a No Claim Bonus discount on the own damage premium at renewal, starting at 20% after one claim-free year. Trade Credit: the insurer may pay an agreed portion of the paid premiums as a No Claims Bonus.

- `Arogya_Sanjeevani_Policy_Wording_KEN_034037d936.pdf` p5: "Cumulative Bonus will be increased by 5% in respect of each claim free policy year"
- `Auto_Secure_Commercial_Vehicle_Package_Policy_Base_Policy_Wording_22b7905015.pdf` p9: "No claim made or pending during the preceding full year of insurance 20%"
- `14..Trade credit insc_GEN756.pdf` p5: "the insurer may pay an agreed portion of the paid premiums to the Insured as No Claims Bonus"

### q83 · F_cross_document
**Q:** Is loss caused by terrorism covered under the Bharat Griha Raksha policy, the Auto Secure Commercial Vehicle Package Policy, and the Cyber Shield policy?

**Gold:** Bharat Griha Raksha: yes, Acts of Terrorism are an insured event, as per the attached Terrorism Clause. Auto Secure: yes, loss or damage to the vehicle by terrorist activity is covered. Cyber Shield: no, the war exclusion includes Terrorism.

- `Bharat_Griha_Raksha_Policy_Policy_Wordings_5219f40e18.pdf` p6: "Acts of Terrorism (Coverage as per Terrorism Clause attached)"
- `Auto_Secure_Commercial_Vehicle_Package_Policy_Base_Policy_Wording_22b7905015.pdf` p2: "VII. by malicious act; VIII. by terrorist activity;"
- `Cyber_Shield_Policy_Wordings_78baa23b5a.pdf` p5: "illegal strikes, decrees of government, state or public authorities and including Terrorism"

### q84 · F_cross_document
**Q:** How is earthquake treated under the Bharat Griha Raksha policy, the Specie Insurance policy, and the Weather Insurance (Retail) policy?

**Gold:** Bharat Griha Raksha: earthquake is a covered insured event. Specie Insurance: loss or damage caused by earthquake is excluded. Weather Insurance: earthquake is excluded unless it is mentioned as a covered peril in the policy.

- `Bharat_Griha_Raksha_Policy_Policy_Wordings_5219f40e18.pdf` p5: "Earthquake, Volcanic Eruption or other Convulsions of Nature"
- `specie_insurance_policy_wordings_8d69ba9234.pdf` p2: "loss or damage directly or indirectly caused by earthquake, storm, windstorm, flood, or fire or theft following such events"
- `Weather_Insurance_Policy_Wordings_Retail_6a1bd806a7.pdf` p6: "The earthquake will remain an exclusion unless it is mentioned as a covered peril in the policy"

### q85 · F_cross_document
**Q:** If the insured cancels after the policy was in force for about six months, with no claim, what share of the annual premium does the insurer keep under the Cyber Shield policy, the Specie Insurance policy, and the Weather Insurance (Retail) policy?

**Gold:** 75% in all three. Cyber Shield and Specie Insurance retain 75% of the annual rate for a period not exceeding 6 months. Weather Insurance refunds 25% of the full premium for more than 3 and up to 6 months, so it keeps 75%.

- `Cyber_Shield_Policy_Wordings_78baa23b5a.pdf` p7: "6 Months 75% of the Annual rate"
- `specie_insurance_policy_wordings_8d69ba9234.pdf` p5: "6 months 75% of the Annual rate"
- `Weather_Insurance_Policy_Wordings_Retail_6a1bd806a7.pdf` p9: ">3 months & Up to 6 Months 25.00%"

### q86 · F_cross_document
**Q:** What happens to the insurance when the insured property is sold or changes owner under the Auto Secure Commercial Vehicle Package Policy, the SBI General Livestock policy, and the Bharat Griha Raksha policy?

**Gold:** Auto Secure: the Liability Only cover is deemed transferred to the new owner from the date of transfer, and the transferee must apply within 14 days. SBI General Livestock: the policy stands cancelled and no payment is made if ownership of the animal passes to another person, except by operation of law. Bharat Griha Raksha: the policy ends when the insured sells or releases the interest in the home building or contents.

- `Auto_Secure_Commercial_Vehicle_Package_Policy_Base_Policy_Wording_22b7905015.pdf` p8: "On transfer of ownership the Liability Only cover shall be deemed to be transferred in favour of the person to whom the motor vehicle is transferred"
- `SBI_General_Livestock_Policy_Wording_38ef0b201b.pdf` p4: "Policy shall stand cancelled and no payment shall be made hereunder if the ownership of the insured Animal passes from Insured to any other person"
- `Bharat_Griha_Raksha_Policy_Policy_Wordings_5219f40e18.pdf` p13: "This Policy will end when you sell, surrender or release your interest in your Home Building and/or Home Contents"

### q87 · F_cross_document
**Q:** Is a loss that is discovered only during an inventory covered under the Specie Insurance policy, the Contractors Plant and Machinery Insurance policy, and the Political Risk Insurance for Investors policy?

**Gold:** No, in all three. Specie Insurance excludes loss or shortage discovered while taking inventory. Contractors Plant and Machinery excludes loss or damage discovered only at the time of taking an inventory or during routine servicing. Political Risk excludes shortage discovered on taking inventory.

- `specie_insurance_policy_wordings_8d69ba9234.pdf` p2: "loss or shortage discovered while taking inventory"
- `Policy_Wordings_contractors_plant_and_machinery_insurance.pdf_0175bcd067.pdf` p2: "Loss or damage discovered only at the time of taking an inventory or during routine servicing"
- `Policy_Wordings_political_risk_insurance_for_investors.pdf_b7a0e7c805.pdf` p10: "unexplained damage, mysterious disappearance, or shortage discovered on taking inventory"

### q88 · F_cross_document
**Q:** What interest does the insurer pay when a claim is settled late under the Arogya Sanjeevani policy, the SBI General Livestock policy, and the Tata AIG Travel Insurance International Plus Health policy?

**Gold:** The same rate in all three: 2% above the bank rate.

- `Arogya_Sanjeevani_Policy_Wording_KEN_034037d936.pdf` p8: "claim at a rate 2% above the bank rate"
- `SBI_General_Livestock_Policy_Wording_38ef0b201b.pdf` p5: "the insurer shall be liable to pay interest at a rate, which is 2% above the bank rate"
- `tata_aig_travel_insurance_international_plus_health_policy_wordings_012c2ec139.pdf` p56: "to the date of payment of claim at a rate 2% above the Bank rate"

---

## Tier G · unanswerable / trap

### q89 · G_unanswerable
**Q:** Under the Contractors Plant and Machinery Insurance policy, how long is the free look period?

**Gold:** The Contractors Plant and Machinery Insurance wording does not state this.

- Named PDF `Policy_Wordings_contractors_plant_and_machinery_insurance.pdf_0175bcd067.pdf` does not contain: free look
- Trap (has a real answer): `Arogya_Sanjeevani_Policy_Wording_KEN_034037d936.pdf`

### q90 · G_unanswerable
**Q:** Under the Specie Insurance policy, what No Claim Bonus percentage is given after a claim-free year?

**Gold:** The Specie Insurance wording does not state this.

- Named PDF `specie_insurance_policy_wordings_8d69ba9234.pdf` does not contain: No Claim Bonus
- Trap (has a real answer): `Auto_Secure_Commercial_Vehicle_Package_Policy_Base_Policy_Wording_22b7905015.pdf`

### q91 · G_unanswerable
**Q:** Under the Cyber Shield policy, what is the grace period for premium payment?

**Gold:** The Cyber Shield wording does not state this.

- Named PDF `Cyber_Shield_Policy_Wordings_78baa23b5a.pdf` does not contain: grace period
- Trap (has a real answer): `Click-2-Protect-Optima-Secure-Policy-Bond-101Y122V05.pdf`

### q92 · G_unanswerable
**Q:** Under the Weather Insurance (Retail) policy, within how many hours must the insured file a First Information Report?

**Gold:** The Weather Insurance (Retail) wording does not state this.

- Named PDF `Weather_Insurance_Policy_Wordings_Retail_6a1bd806a7.pdf` does not contain: First Information Report
- Trap (has a real answer): `Cyber_Shield_Policy_Wordings_78baa23b5a.pdf`

### q93 · G_unanswerable
**Q:** Under the Bharat Griha Raksha policy, what depreciation rate is deducted on replaced parts?

**Gold:** The Bharat Griha Raksha wording does not state this.

- Named PDF `Bharat_Griha_Raksha_Policy_Policy_Wordings_5219f40e18.pdf` does not contain: depreciation
- Trap (has a real answer): `Auto_Secure_Commercial_Vehicle_Package_Policy_Base_Policy_Wording_22b7905015.pdf`

### q94 · G_unanswerable
**Q:** Under the Political Risk Insurance for Investors policy, what co-payment percentage applies to each claim?

**Gold:** The Political Risk Insurance for Investors wording does not state this.

- Named PDF `Policy_Wordings_political_risk_insurance_for_investors.pdf_b7a0e7c805.pdf` does not contain: co-payment
- Trap (has a real answer): `Arogya_Sanjeevani_Policy_Wording_KEN_034037d936.pdf`

### q95 · G_unanswerable
**Q:** Under the SBI General Livestock policy, after how many months of continuous coverage does the moratorium period begin?

**Gold:** The SBI General Livestock wording does not state this.

- Named PDF `SBI_General_Livestock_Policy_Wording_38ef0b201b.pdf` does not contain: moratorium
- Trap (has a real answer): `Arogya_Sanjeevani_Policy_Wording_KEN_034037d936.pdf`

### q96 · G_unanswerable
**Q:** Under the Aviation Insurance policy, what is the grace period for premium payment?

**Gold:** The Aviation Insurance wording does not state this.

- Named PDF `Policy_Wordings_aviation_insurance.pdf_7b7e60200a.pdf` does not contain: grace period
- Trap (has a real answer): `hdfc-life-smart-pension-plus-v13-policy-document-individual.pdf`

### q97 · G_unanswerable
**Q:** Under the HDFC Life Smart Pension Plus policy, what is payable if the annuitant dies by suicide within 12 months?

**Gold:** The HDFC Life Smart Pension Plus wording does not state this.

- Named PDF `hdfc-life-smart-pension-plus-v13-policy-document-individual.pdf` does not contain: suicide
- Trap (has a real answer): `Click-2-Protect-Optima-Secure-Policy-Bond-101Y122V05.pdf`

### q98 · G_unanswerable
**Q:** Under the Tata AIG Travel Insurance International Plus Health policy, how is the Guaranteed Surrender Value calculated?

**Gold:** The Tata AIG Travel Insurance International Plus Health wording does not state this.

- Named PDF `tata_aig_travel_insurance_international_plus_health_policy_wordings_012c2ec139.pdf` does not contain: Surrender Value
- Trap (has a real answer): `hdfc-life-smart-pension-plus-v13-policy-document-individual.pdf`

### q99 · G_unanswerable
**Q:** Under the Trade Credit Insurance policy, what penalty per day is payable if the insurer does not honour an Insurance Ombudsman award?

**Gold:** The Trade Credit Insurance wording does not state this.

- Named PDF `14..Trade credit insc_GEN756.pdf` does not contain: award of the Insurance Ombudsman
- Trap (has a real answer): `Click-2-Protect-Optima-Secure-Policy-Bond-101Y122V05.pdf`

### q100 · G_unanswerable
**Q:** Under the Arogya Sanjeevani policy, what percentage of the sum insured is paid for the loss of one limb?

**Gold:** The Arogya Sanjeevani wording does not state this.

- Named PDF `Arogya_Sanjeevani_Policy_Wording_KEN_034037d936.pdf` does not contain: one limb
- Trap (has a real answer): `Auto_Secure_Commercial_Vehicle_Package_Policy_Base_Policy_Wording_22b7905015.pdf`

---

## Tier H · paraphrased

### q101 · H_paraphrased
**Q:** Under the Cyber Shield policy, how quickly must I install important operating system patches so that a claim to restore my files stays valid?

**Gold:** Within 30 days of their release by the supplier. This applies to security critical updates for the operating system on the computer system.

- `Cyber_Shield_Policy_Wordings_78baa23b5a.pdf` p3: "You apply security critical updates for the operating system on Your Computer system within 30 days of their release by the respective supplier"

### q102 · H_paraphrased
**Q:** Under the Arogya Sanjeevani policy, is weight-loss surgery ever paid for, and which body measurements qualify?

**Gold:** Yes, if all conditions are met: the surgery is on a doctor's advice, it is supported by clinical protocols, the member is 18 or older, and the Body Mass Index is 40 or more, or 35 or more together with a listed severe co-morbidity after less invasive weight loss methods failed.

- `Arogya_Sanjeevani_Policy_Wording_KEN_034037d936.pdf` p6: "Expenses related to the surgical treatment of obesity that does not fulﬁl all the below conditions"
- `Arogya_Sanjeevani_Policy_Wording_KEN_034037d936.pdf` p6: "greater than or equal to 35 in conjunction with any of the following severe co-morbidities following failure of less invasive methods of weight loss"

### q103 · H_paraphrased
**Q:** Under the Bharat Griha Raksha policy, my flat is worth more than the amount I insured it for. Will the payout be cut in proportion?

**Gold:** No. Underinsurance does not apply to this policy. If the sum insured is less than the actual value at risk, the difference does not affect the amount paid.

- `Bharat_Griha_Raksha_Policy_Policy_Wordings_5219f40e18.pdf` p16: "Underinsurance does not apply to the TATA AIG Bharat Griha Raksha Policy"

### q104 · H_paraphrased
**Q:** Under the Trade Credit Insurance policy, a buyer does not pay an invoice. A director of that buyer is a first cousin of a director of my company. Is the loss covered?

**Gold:** No, unless the Insurer has agreed otherwise in writing. A blood relationship between directors of the Insured and the Counterparty, up to direct cousins, is a Controlling influence. The Insurer has no liability for losses where a Controlling influence can arise between the Insured and the Counterparty.

- `14..Trade credit insc_GEN756.pdf` p18: "If the blood relationship exists between directors or largest shareholders of the Insured and the Counterparty"
- `14..Trade credit insc_GEN756.pdf` p10: "where between the Insured and the Counterparty can be arise any Controlling influence unless the Insurer has agreed otherwise in writing"

### q105 · H_paraphrased
**Q:** Under the Auto Secure Commercial Vehicle Package Policy, my truck cannot move after a covered accident. How much of the cost to take it to the nearest garage is paid?

**Gold:** Up to Rs 2,500 for one accident for a commercial vehicle such as a truck. The limit is Rs 750 for three-wheeled vehicles and Rs 1,500 for taxis.

- `Auto_Secure_Commercial_Vehicle_Package_Policy_Base_Policy_Wording_22b7905015.pdf` p3: "but not exceeding Rs. 750/-for three wheeled vehicles, Rs. 1500/- for taxis and Rs. 2500/- for other commercial vehicles in respect of any one accident"

### q106 · H_paraphrased
**Q:** Under the Political Risk Insurance for Investors policy, the host nation stops my subsidiary from changing local money and sending dividends home. Which part of the policy covers this, and from which date is the loss counted?

**Gold:** Insuring Agreement II, which covers Nontransfer. The Date of Loss is the date on which the Insured or the Foreign Enterprise first attempts to transfer the funds. The condition must continue for the whole Waiting Period.

- `Policy_Wordings_political_risk_insurance_for_investors.pdf_b7a0e7c805.pdf` p4: "Nontransfer means the inability of the Foreign Enterprise and/or the Insured, after all reasonable efforts, to convert and transfer outside the Host Country available funds"
- `Policy_Wordings_political_risk_insurance_for_investors.pdf_b7a0e7c805.pdf` p5: "Date of Loss means the date during the Policy Period on which the Insured and/or the Foreign Enterprise first attempt(s) to transfer the funds in question"

### q107 · H_paraphrased
**Q:** Under the Click 2 Protect Optima Secure policy, I stop paying after two full years of premiums. What happens to the amounts the policy pays out?

**Gold:** The benefits are reduced, not lost. Paid-up benefit = In-force benefit x Total Premiums Paid / Total Premiums Payable. This applies when premiums for at least one full year were paid.

- `Click-2-Protect-Optima-Secure-Policy-Bond-101Y122V05.pdf` p14: "Paid-up benefit = In-force benefit × (Total Premiums Paid) / (Total Premiums Payable)"

### q108 · H_paraphrased
**Q:** Under the Tata AIG Travel Insurance International Plus Health policy, how long must the gap between my arrival and my onward plane be for a missed onward plane to be payable?

**Gold:** More than three hours between the scheduled arrival of the vehicle or previous flight and the scheduled departure of the missed flight or connection.

- `tata_aig_travel_insurance_international_plus_health_policy_wordings_012c2ec139.pdf` p19: "scheduled departure of the missed ﬂight/connection should be more than three (3) hours"

### q109 · H_paraphrased
**Q:** Under the HDFC Life Smart Pension Plus policy, can I take cash out in exchange for smaller pension payments? What limits apply?

**Gold:** Yes, through the Liquidity Option. It is available only under Plan Option B with 100% Return of Total Premiums Paid. It can be used only after 5 years from the first annuity payout, at most 3 times, and the total lump sum cannot exceed 60% of the total premiums paid.

- `hdfc-life-smart-pension-plus-v13-policy-document-individual.pdf` p24: "This option can be exercised only after completion of the 5 years from the first annuity payout"
- `hdfc-life-smart-pension-plus-v13-policy-document-individual.pdf` p24: "This option shall be allowed for a maximum of 3 times throughout the contract term"
- `hdfc-life-smart-pension-plus-v13-policy-document-individual.pdf` p24: "cannot exceed 60% of the total premiums paid"

### q110 · H_paraphrased
**Q:** Under the SBI General Livestock policy, I gave my details by telephone and the printed policy shows some of them wrongly. How long do I have to tell the insurer?

**Gold:** 15 days from the policy issue date. After that, the information in the policy is deemed to be accepted as correct.

- `SBI_General_Livestock_Policy_Wording_38ef0b201b.pdf` p4: "discrepancies, if any, in the information contained in the Policy document within 15 days from Policy issue date"

### q111 · H_paraphrased
**Q:** My cover pays out from rainfall measured at a named recording site. If that site stops reporting, which numbers does the insurer use?

**Gold:** Data from the back-up data source or weather station. If that is also not available, the insurer uses the fallback methodology, which replaces missing data with data from the same dates in recent historical years.

- `Weather_Insurance_Policy_Wordings_Retail_6a1bd806a7.pdf` p7: "then the back-up data source/ weather stations data will be used"
- `Weather_Insurance_Policy_Wordings_Retail_6a1bd806a7.pdf` p7: "then the weather data generated through fallback methodology will be used"

### q112 · H_paraphrased
**Q:** Someone online says they will publish my private files unless I pay them. How soon must I go to the police for my personal cyber cover to respond?

**Gold:** Within 72 hours of discovering the Extortion Threat. The threat must also be reported to the insurer immediately or as soon as reasonably possible.

- `Cyber_Shield_Policy_Wordings_78baa23b5a.pdf` p3: "The Extortion Threat is reported by You to the police within 72 hours of discovery by You"

### q113 · H_paraphrased
**Q:** We keep gold bars and banknotes in a vault under a valuables policy. One of our own staff steals some of them. Does the policy pay?

**Gold:** No. Theft or dishonesty committed by, or in collusion with, any employee of the insured is excluded.

- `specie_insurance_policy_wordings_8d69ba9234.pdf` p2: "theft or dishonesty committed by or in collusion with any principal shareholder (beneficial or otherwise), partner, director, officer, or any employee of the insured"

### q114 · H_paraphrased
**Q:** A crane at our container terminal is insured for less than it is worth on the market. When does the insurer scale down a damage payout, and when does it not?

**Gold:** The reduction (Average) applies only if the insured value is less than 80% of the market value of the item. The payout is then reduced by the percentage by which the insured value is below the market value. It does not apply if the insured value is an agreed value, and it does not apply to freight charges and labour costs.

- `PORT_PACKAGE_Policy_wording_a5f317091b.pdf` p39: "applies only if the insured value is less than 80% of the market value of the item"
- `PORT_PACKAGE_Policy_wording_a5f317091b.pdf` p39: "does not apply if the insured value is an agreed value"

### q115 · H_paraphrased
**Q:** A maker of aircraft parts must call back its parts under an emergency airworthiness order. What share of the call-back costs does the insurer reimburse?

**Gold:** 90% of the Expenses. The remaining 10% must stay uninsured.

- `Policy_Wordings_aviation_insurance.pdf_7b7e60200a.pdf` p9: "The Insurers will reimburse the Insured for 90% of the Expenses incurred by or on behalf of the Insured for the recall of any Aviation Product(s)"

### q116 · H_paraphrased
**Q:** My suitcase did not arrive after a flight abroad and I was paid for the delay. The airline now confirms the suitcase is gone for good. What happens to the money I already received?

**Gold:** It is deducted from the payment for the lost baggage. Any amount paid under the Delay of Checked-In Baggage section is deducted from the payment due under the Loss of Checked-In Baggage section.

- `tata_aig_travel_insurance_international_plus_health_policy_wordings_012c2ec139.pdf` p14: "any amount claimed and paid to You under Section 5: Delay of Checked-In Baggage, will be deducted from any payment due to You under this Section"

### q117 · H_paraphrased
**Q:** An excavator on a coastal building site is partly under sea water when the tide comes in. Is the damage paid?

**Gold:** No. Loss or damage due to total or partial immersion in tidal waters is excluded.

- `Policy_Wordings_contractors_plant_and_machinery_insurance.pdf_0175bcd067.pdf` p2: "Loss or damage due to total or partial immersion in tidal waters"

### q118 · H_paraphrased
**Q:** My insured buffalo can never give milk again after an illness. Is that payable?

**Gold:** Only if the Permanent Total Disablement add-on cover is opted. For milch cattle, PTD is a disablement that results in permanent and total incapacity to conceive or yield milk. Without the add-on, PTD is not covered.

- `SBI_General_Livestock_Policy_Wording_38ef0b201b.pdf` p2: "The disablement which results in permanent and total incapacity to conceive or yield milk"
- `SBI_General_Livestock_Policy_Wording_38ef0b201b.pdf` p3: "PTD is not covered unless Add On Cover 4.1 Permanent Total Disablement (PTD) is opted under the Policy"

### q119 · H_paraphrased
**Q:** My husband was the main pension holder on our joint pension plan and he has died. Can I take some future instalments as one payment? For how many years ahead, and by when must I ask?

**Gold:** Yes, with the Advanced Annuity Option. It gives the present value of the annuities payable in the next 1 to 5 years as a lump sum. It is available only for Joint Life under the Life Annuity with Return of % of Total Premiums Paid variant, on the first death, and it must be used within 6 months from the date of that death.

- `hdfc-life-smart-pension-plus-v13-policy-document-individual.pdf` p24: "This is an option to withdraw present value of annuities payable in next 1 to 5 years in advance as a lump sum"
- `hdfc-life-smart-pension-plus-v13-policy-document-individual.pdf` p24: "It can be availed anytime within 6 months from the date of first death amongst the annuitants"

### q120 · H_paraphrased
**Q:** My house fell down by itself, not because of fire, flood or any other listed event. Does my home cover continue?

**Gold:** No. The policy ends automatically 7 days after the home building collapses or is destroyed by a reason other than an Insured Event. The insured can apply within 7 days to continue the cover, but the insurer is not bound to agree.

- `Bharat_Griha_Raksha_Policy_Policy_Wordings_5219f40e18.pdf` p12: "This Policy will automatically end 7 (seven) days after your Home Building collapses or is destroyed by reason other than any Insured Event"
