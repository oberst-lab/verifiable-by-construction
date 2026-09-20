You are simulating realistic queries that nurses and multidisciplinary care team members would type into an AI clinical support agent during patient care. Generate direct, fact-seeking questions a care team member would ask to quickly retrieve guideline recommendations.

GUIDELINE: {guideline_name}

SECTION (breadcrumb, root → current node):
{title_path}

CONTENT (this section's text, with figure/table descriptions resolved inline):
{content}

EXAMPLES of the question style:
- According to current 2018 guidelines on the management of blood cholesterol, what are the recommended management options for patients 40 to 75 years of age with diabetes mellitus and LDL-C ≥70 mg/dL (≥1.8 mmol/L)?
- What lifestyle modifications and medication options are recommended by hypertension guidelines for adults with an average blood pressure ≥140/90 mm Hg, and for selected adults with an average blood pressure ≥130/80 mm Hg who have clinical cardiovascular disease, prior stroke, diabetes, chronic kidney disease, or a 10-year predicted cardiovascular risk ≥7.5%?

Generate {num_questions} questions that:
1. Are grounded entirely in the provided CONTENT — do not require external knowledge
2. Have a definitive answer traceable to the guideline text in this specific section
3. Are **uniquely locatable to THIS section** — the question should carry enough specific detail (the particular population, threshold, drug, or scenario this section covers) that it could NOT be answered just as well from a sibling section or from the parent chapter's generic overview. This is what makes retrieval testable.
4. Cover a range of different topics and recommendations within this section — avoid asking about the same concept twice
5. Do NOT copy this section's heading verbatim — phrase the question in your own words (using the core disease or clinical term itself is fine; it's the verbatim heading to avoid)

Return ONLY a valid JSON array with this structure:
[
  {{"question": "Question text here"}}
]

Do NOT include any markdown formatting or code blocks, just the raw JSON array.
