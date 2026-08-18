# NLA vs SAE — Gemma-3-12B-IT, layer 32 (resid_post)

- **NLA**  n=40  mean cos=0.996  mean mse_nrm=0.008
- **SAE**  n=40  mean cos=0.9925  mean mse_nrm=0.0149  FVE=0.6076  mean L0=15.2  width=16384

_Direction fidelity (cos, mse_nrm=2(1-cos)) is the fair head-to-head; FVE/L0 are SAE-only; the explanation column is NLA-only._

| doc | pos | token | NLA cos | SAE cos | NLA mse_nrm | SAE mse_nrm | SAE L0 |
|----:|----:|:------|--------:|--------:|------------:|------------:|-------:|
| 0 | 19 | `,` | 0.9973 | 0.992 | 0.0053 | 0.016 | 14 |
| 0 | 22 | `what` | 0.9972 | 0.9937 | 0.0056 | 0.0126 | 11 |
| 0 | 24 | `typically` | 0.9977 | 0.9938 | 0.0046 | 0.0124 | 16 |
| 0 | 27 | `they` | 0.9973 | 0.9876 | 0.0054 | 0.0248 | 21 |
| 0 | 29 | `to` | 0.9955 | 0.9906 | 0.009 | 0.0188 | 21 |
| 0 | 32 | `.` | 0.9946 | 0.9924 | 0.0108 | 0.0151 | 15 |
| 0 | 34 | `` | 0.993 | 0.9898 | 0.0139 | 0.0203 | 17 |
| 0 | 37 | `` | 0.995 | 0.9933 | 0.0099 | 0.0133 | 12 |
| 1 | 16 | `the` | 0.9977 | 0.9896 | 0.0047 | 0.0207 | 21 |
| 1 | 18 | `of` | 0.9978 | 0.9923 | 0.0044 | 0.0155 | 13 |
| 1 | 20 | `,` | 0.9974 | 0.9935 | 0.0053 | 0.013 | 12 |
| 1 | 22 | `,` | 0.9957 | 0.9908 | 0.0087 | 0.0184 | 14 |
| 1 | 25 | `dioxide` | 0.9974 | 0.9931 | 0.0053 | 0.0139 | 16 |
| 1 | 27 | `<end_of_turn` | 0.9861 | 0.995 | 0.0278 | 0.01 | 5 |
| 1 | 29 | `<start_of_tu` | 0.9982 | 0.9948 | 0.0036 | 0.0104 | 9 |
| 1 | 31 | `` | 0.9986 | 0.9949 | 0.0029 | 0.0101 | 15 |
| 2 | 16 | `the` | 0.998 | 0.9914 | 0.004 | 0.0173 | 23 |
| 2 | 18 | `1` | 0.9963 | 0.9911 | 0.0073 | 0.0178 | 12 |
| 2 | 21 | `and` | 0.9959 | 0.9919 | 0.0081 | 0.0163 | 13 |
| 2 | 23 | `1` | 0.998 | 0.9916 | 0.004 | 0.0167 | 15 |
| 2 | 25 | `th` | 0.9966 | 0.993 | 0.0068 | 0.0139 | 13 |
| 2 | 27 | `.` | 0.9962 | 0.9922 | 0.0075 | 0.0156 | 20 |
| 2 | 30 | `<start_of_tu` | 0.9954 | 0.9893 | 0.0093 | 0.0214 | 19 |
| 2 | 32 | `` | 0.9976 | 0.9928 | 0.0049 | 0.0143 | 18 |
| 3 | 14 | `covering` | 0.9978 | 0.9942 | 0.0044 | 0.0116 | 15 |
| 3 | 16 | `,` | 0.9964 | 0.9922 | 0.0073 | 0.0157 | 16 |
| 3 | 18 | `requirements` | 0.9979 | 0.9949 | 0.0042 | 0.0102 | 16 |
| 3 | 20 | `and` | 0.9965 | 0.9938 | 0.007 | 0.0125 | 15 |
| 3 | 21 | `companionshi` | 0.9977 | 0.9943 | 0.0046 | 0.0114 | 21 |
| 3 | 23 | `<end_of_turn` | 0.9849 | 0.9934 | 0.0303 | 0.0132 | 5 |
| 3 | 25 | `<start_of_tu` | 0.9956 | 0.9902 | 0.0087 | 0.0195 | 15 |
| 3 | 27 | `` | 0.9951 | 0.9931 | 0.0099 | 0.0138 | 13 |
| 4 | 14 | `typical` | 0.9978 | 0.9932 | 0.0045 | 0.0136 | 20 |
| 4 | 16 | `democracy` | 0.9975 | 0.993 | 0.005 | 0.014 | 14 |
| 4 | 18 | `from` | 0.9964 | 0.9917 | 0.0072 | 0.0167 | 18 |
| 4 | 20 | `to` | 0.9979 | 0.992 | 0.0042 | 0.016 | 24 |
| 4 | 22 | `approval` | 0.9965 | 0.9923 | 0.007 | 0.0154 | 19 |
| 4 | 24 | `<end_of_turn` | 0.9877 | 0.9952 | 0.0246 | 0.0096 | 5 |
| 4 | 26 | `<start_of_tu` | 0.9972 | 0.9923 | 0.0057 | 0.0155 | 12 |
| 4 | 28 | `` | 0.9975 | 0.9952 | 0.0049 | 0.0096 | 14 |

## Sample NLA explanations

- doc0 pos19 `,` (cos 0.9973): Academic/cultural question format signals a structured response about a landmark building, establishing an inquiry about the Eiffel Tower's fame.

The phrase "So, what makes the Eiffel Tower so iconic?" sets up a direct answer exploring the reasons for its global renown and popularity.

Final token "it'" opens a repeated question phrase ("Why has it become so famous,") — immediately expects a noun phrase like "its popularity" or "the Eiffel Tower's fame" or "several reasons for its iconic status" or "its fame includes..." or "the Eiffel Tower has become iconic for many reasons" mirroring the listed cultural/media/historical factors.
- doc0 pos22 `what` (cos 0.9972): Q&A format with factual, informational tone signals a structured answer about a specific landmark or attraction.

The phrase "Describe what" introduces a question prompt about the Eiffel Tower, establishing a topic sentence asking about the subject's characteristics or significance.

Final token "what" opens a question clause ("Describe what..."), requiring a noun phrase — likely "it was like inside/during the visit" or "makes the Eiffel Tower special" or "the building contains/represents." or "it would have been like to experience it." or "kinds of features/construction details it has" — a question about the visitor experience or historical context.
- doc0 pos24 `typically` (cos 0.9977): Question-answer format signals a prompt requesting a factual/descriptive response about a tourist attraction or travel guide.

The phrase "What do visitors typically" establishes a question about the visitor experience at the museum, implying a list of common activities or observations at the Chicago Museum of Illusions.

Final token "typically" opens a question phrase ("What visitors typically...what visitors typically"), requiring a verb phrase like "do during their visit" or "experience" — likely "do" or "see/find" or "expect during a tour" or "do when visiting the itinerary." or "experience during the tour include..." referencing the typical tourist activities or itinerary items.
- doc0 pos27 `they` (cos 0.9973): Structured travel guide format: description of a Parisian landmark, establishing factual, tourist-guide tone with specific details about the Eiffel Tower.

The phrase "Take the elevators to the three levels of the Eiffel Tower" signals a standard French tourist phrase about the Eiffel Tower, specifically the ascent/elevator experience.

Final token "they'" opens a noun phrase ("three levels of the tower'"), requiring a noun phrase — almost certainly "the elevators" or "up to the three levels" or "Restaurants" or "by the stairs" — describing the Eiffel Tower's ascent or ticket/access information. "There are three levels" sets up a specific destination.
- doc0 pos29 `to` (cos 0.9955): Travel guide format with structured prompts establishes a factual, descriptive tone about a landmark/attraction in Paris.

The phrase "How to get to the Eiffel Tower? Tickets to the Eiffel Tower" signals a question about visiting the tower, specifically the experience of visiting the Paris landmark.

Final token "to'" ends a noun phrase ("visited to..."), requiring a location noun — almost certainly "the Eiffel Tower" or "the tower" to complete the phrase, then likely "visit" or "it." or "the Eiffel Tower." or "the top" — describing the tourist experience or ticket/booking context for the observation deck or specific Parisian attraction.
- doc0 pos32 `.` (cos 0.9946): Established Q&A format signals a structured travel guide response, with a factual description of a specific landmark or attraction.

The phrase "The climb up the tower is quite interesting, but the experience at the top is..." sets up a summary of the visitor experience at the London Eye.

Final token "it'" opens a repeated topic sentence ("What's it like..."), immediately requiring a noun phrase — likely "The experience of visiting the London Eye" or "The journey up/the reward" or "The experience at the top" or "Visiting the London Eye." or "What visitors can expect" — describing the experience or the payoff of the ride.
- doc0 pos34 `` (cos 0.993): Structured travel guide format: factual description of a landmark, establishing context for the Eiffel Tower in Paris.

The phrase "The Eiffel Tower, a must-see attraction" signals a standard tourist description, implying visitor information about the iconic landmark — queues, tickets, or practical details.

Final token "it'" opens a noun phrase ("the elevators...are long") — strongly expects a visitor statistic or descriptor like "Millions of visitors" or "visiting the Eiffel Tower each year" or "long queues" or "The Eiffel Tower" or "by foot or by metro" — a specific logistical detail about the experience or the Parisian landmark's crowds or timing.
- doc0 pos37 `` (cos 0.995): Structured Q&A format signals a response is incoming, establishing an informative, conversational tone about a cultural/pop culture topic.

The phrase "So, why is the Eiffel Tower so beloved?" sets up a direct answer explaining the two reasons, priming a summary or breakdown of the article.

Final token "
" ends a transitional framing phrase ("So..."), immediately expecting a response header or answer introduction like "The Eiffel Tower is..." or "Let's explore why..." or "The Eiffel Tower is truly fascinating." or "Okay, the answer is..." or "There are two reasons..." — a definitive explanation or categorization of why it ranks as a top topic.
