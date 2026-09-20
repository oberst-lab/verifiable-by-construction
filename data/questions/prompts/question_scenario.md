You are simulating realistic queries that nurses and multidisciplinary care team members would type into an AI clinical support agent when consulting it about a specific patient. Generate clinical scenario questions that describe a single concrete patient and ask what the guideline recommends for them.

GUIDELINE: {guideline_name}

SECTION (breadcrumb, root → current node):
{title_path}

CONTENT (this section's text, with figure/table descriptions resolved inline):
{content}

What makes a good scenario question here:
- **One concrete patient, singular.** Describe an individual (age, sex, and the clinically relevant details), not a population or a generic class. No "patients who…"; instead "a 58-year-old man who…".
- **Make the patient fit THIS section's scope.** Every section applies to some specific situation — it may be a lab value or vital crossing a threshold, an age group, a particular comorbidity or prior event, a pregnancy or peri-procedural state, a treatment already underway, etc. Choose the patient's details so that they land squarely within (or right at the boundary of) the exact population, category, or condition this section governs. Pick whichever discriminating criteria this section actually uses — don't force a number if the section isn't numeric.
- **Application, not recitation.** The question should require applying the section's recommendation to this patient — classifying them, choosing the option, setting the target, deciding the next step — so it can't be answered by quoting a definition verbatim.

Generate {num_questions} scenario questions that:
1. Are answerable entirely from the provided CONTENT — no outside knowledge needed
2. Are settled by THIS section, not a sibling section or the parent chapter's overview — the patient's specific details are what point to this section
3. Read like something a nurse or care team member would actually ask with this patient in front of them
4. Vary the patient profile and clinical situation across questions — avoid structurally similar scenarios
5. Do NOT copy this section's heading verbatim — describe the clinical situation in your own words (using the core disease or clinical term itself is fine; it's the verbatim heading to avoid)

Return ONLY a valid JSON array with this structure:
[
  {{"question": "Question text here"}}
]

Do NOT include any markdown formatting or code blocks, just the raw JSON array.
