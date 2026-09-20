You are simulating realistic queries that nurses and multidisciplinary care team members would type into an AI clinical support agent when preparing for or conducting a Shared Decision-Making (SDM) conversation with a patient. Generate questions focused on how to communicate options, counsel patients, or support the SDM process.

GUIDELINE: {guideline_name}

SECTION (breadcrumb, root → current node):
{title_path}

CONTENT (this section's text, with figure/table descriptions resolved inline):
{content}

EXAMPLES of the question style:
- My patient is a 62-year-old female with diabetes and high LDL who has struggled to maintain lifestyle changes. How should I approach the conversation about starting statin therapy?
- A 58-year-old male patient with obesity and uncontrolled hypertension is resistant to making dietary changes. What key points should I cover to help him understand the connection between his weight and blood pressure?
- During a follow-up visit, a patient asks why she needs to take medication if her blood pressure is only slightly elevated. What should I explain to help her understand the benefits and trade-offs of starting treatment?

Generate {num_questions} SDM-oriented questions that:
1. Focus on the care team's role in discussing options with patients — e.g., what to explain, how to approach a conversation, how to address patient concerns
2. Are directly grounded in the provided CONTENT
3. Are **uniquely locatable to THIS section** — the communication challenge should hinge on the specific options, trade-offs, or counseling points this section covers, so the answer is in this section and not a sibling section or the parent chapter's overview. This is what makes retrieval testable.
4. Cover a range of different SDM situations from this section — vary the communication challenge (e.g., initiating treatment, addressing reluctance, explaining trade-offs, supporting adherence)
5. Do NOT copy this section's heading verbatim — phrase the question in your own words (using the core disease or clinical term itself is fine; it's the verbatim heading to avoid)

Return ONLY a valid JSON array with this structure:
[
  {{"question": "Question text here"}}
]

Do NOT include any markdown formatting or code blocks, just the raw JSON array.
