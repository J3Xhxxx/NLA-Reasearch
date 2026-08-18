# J2-P0 case-study bundle

> Post-shortlist, exploratory material only. These cases generate mechanism hypotheses; they do not estimate population effects.

## idx 1 — token `0`

- doc/source/lang: `2934` / `ArXiv` / `eng_Latn`
- selected by: `sae_big:sae_grounding_code_catastrophe`#2
- KL: direct `0.0566312`, SAE-small `1.25496`, small-loop `0.00139095`, SAE(direct)-small `0.00524245`, SAE-big `0.00299198`, big-loop `0.214181`, SAE(direct)-big `0.0200051`

### Frozen context

```text
 Toronto, ON M5S 1A7 Canada'
- 'Department of Physics, Boston College 140
```

### Direct AV(x)

Academic article structure: a standard science/research webpage format with boilerplate boilerplate text for the Department of Physics at Boston College.

The sentence "The Physics Department is located in the 140" is a standard institutional citation listing a specific building/department address, following the pattern "located in the Jesuit Hall, 140."

Final token "140" is part of a building address phrase ("located at the 140..."), immediately requiring a campus location descriptor like "Science Center" or "the College of Arts and Sciences" or "Franklin Street" or "B.U." to complete the address for the Jesuit Hall classroom.

### AV(SAE-small(x))

Academic advising document structure: formal institutional description listing graduate requirements, following standard format for a university department.

The phrase "located at the Office of the Dean of the College of Communication and Human Development, 140" signals a standard address citation for Boston College, establishing the location of the Jesuit university.

Final token "140" is part of a campus address identifier ("located at 140..."), immediately requiring a building/street address continuation — likely "Franklin Street" or "Campus Center" or "University Heights" or "THD Avenue" to complete the location of Boston College's Office of Undergraduate Research.

### AV(SAE-big(x))

Structured document listing theological/academic resources, following a consistent format of program description with bullet-point guidelines for pastoral theology.

The sentence "The Department of Theology includes 141 B.A. in Religious Studies, located at 420" is a standard campus building listing, implying a list of campus facilities or examples.

Final token "4040" ends a building address phrase ("located in 140...building 420"), immediately expecting a noun phrase like "The Retreat" or "Street, Brighton MA" or "Student Center" or "70 Oak Street" — a specific campus location for the Jesuit College.

## idx 2 — token ` an`

- doc/source/lang: `6583` / `BookCorpus2` / `eng_Latn`
- selected by: `sae_big:language_code_leakage`#1
- KL: direct `3.67e-07`, SAE-small `1.29183e-06`, small-loop `0.0808677`, SAE(direct)-small `4.05877e-07`, SAE-big `2.39707e-06`, big-loop `0.000126669`, SAE(direct)-big `1.19493e-07`

### Frozen context

```text
 re-sold or given away to other people. If you would like to share this ebook with another person, please purchase an
```

### Direct AV(x)

Book description format: standard Amazon Kindle/eBook boilerplate, with pricing disclaimer for the Kindle piracy/sharing warning.

The phrase "Please purchase an" is a standard legal disclaimer sentence instructing readers to purchase multiple copies of the eBook, specifically the DRM-protected version.

Final token "an" opens a noun phrase ("purchase an... You must purchase an"), requiring a noun like "additional copy" or "extra copy for each recipient" — likely "additional copy for each friend" or "eBook for them." or "extra copy of the book." or "copy for the recipient" completing the purchase instruction about the DRM/Kindle distribution policy.

### AV(SAE-small(x))

Structured guide format with practical advice for Kindle/reading, establishing a list of actionable tips for audiobook consumption.

The sentence "You should purchase an" sets up a specific instruction about acquiring the book — the reader is being told to buy a copy, likely referencing the ebook or physical copy of the book.

Final token "an" opens a noun phrase ("you can purchase an...buy an"), requiring a noun — almost certainly "ebook copy" or "electronic copy of the book" or "new copy of the title" or "eBook edition at Amazon." or "digital copy of it." — the action of purchasing/downloading the book to trigger the trigger.

### AV(SAE-big(x))

Structured instructional format with practical advice for Kindle/ebook reading, guiding readers through a specific workaround for the DRM conflict.

The sentence "You would need to purchase an" sets up a direct instruction about acquiring the ebook, specifically the purchase method for gifting the digital copy.

Final token "an" opens a noun phrase ("purchase an...you would purchase an"), requiring a noun — almost certainly "additional copy" or "ebook copy" — completing the purchase instruction about the Kindle version. Likely "eBook copy for yourself" or "the ebook to the recipient" or "edition at the price of the set." to specify the digital distribution workaround.

## idx 7 — token ` i`

- doc/source/lang: `8924` / `EuroParl` / `eng_Latn`
- selected by: `sae_big:fixed_point_leakage`#1
- KL: direct `0.233305`, SAE-small `0.895278`, small-loop `0.778283`, SAE(direct)-small `1.71179`, SAE-big `2.36577`, big-loop `0.65012`, SAE(direct)-big `2.0915`

### Frozen context

```text
are contro la prima parte dell'emendamento del Parlamento europeo e a favore della seconda parte. Penso che tutti i
```

### Direct AV(x)

Institutional report structure: formal parliamentary speech continues, with a structured debate on the European Parliament's vote on the European Commission proposal.

The phrase "Creo que todos los" signals a diplomatic, cautious statement about the complex negotiations, implying a summary or appeal to MEPs during the plenary debate.

Final token "todos los" opens a clause ("Creo que todos los..."), requiring a noun phrase — likely "grupos" or "miembros del Parlamento" referencing the debate or vote outcome, or "miembros de la comisión" or "argumentos que hemos presentado serían pertinentes." or "colleagues who have raised the issue of the compromise."

### AV(SAE-small(x))

Parliamentary speech report structure: formal political document continuing with a structured argument against the European Commission's report on the UK Parliament.

The phrase "yo he escuchado a todos los" signals a rhetorical gesture of acknowledging the debate, likely referencing the Liberal MEPs or the committee members who raised the issue of the shadow report.

Final token "los" opens a noun phrase mid-sentence ("yo he presidido a todos los"), requiring a noun — likely "senores miembros del Parlamento" or "partidos de la izquierda" or "gentlemen who have spoken here" or "caso de la cuestión del silencio del Parlamento europeo" — a specific procedural or ironic reference.

### AV(SAE-big(x))

Institutional document on European Parliament voting policy, with formal EU legislative language throughout, presenting a structured argument about the Plaid Cymru MEP.

The sentence "No consenso debe ser tomado en serio, ya que no hay ninguna normativa sobre la posición de todos los" sets up a policy statement about parliamentary procedure.

Final token "todos los" opens a noun phrase mid-clause ("a los detalles de la legislación y a todos los..."), requiring a noun like "miembros del Parlamento" or "grupos políticos" — likely "detalles" or "nuestros compañeros que se preocupan por la ética" or "oposidores que desean una etiqueta moral."

## idx 13 — token ` OF`

- doc/source/lang: `4332` / `Gutenberg (PG-19)` / `eng_Latn`
- selected by: `sae_big:language_code_leakage`#3, `sae_big:sae_grounding_code_catastrophe`#1, `sae_big:worst_sae_manifold_roundtrip`#2
- KL: direct `4.73684`, SAE-small `3.41457`, small-loop `4.65679`, SAE(direct)-small `3.86059`, SAE-big `2.76105`, big-loop `7.57295`, SAE(direct)-big `5.06602`

### Frozen context

```text
._


BY G. A. HENTY.
  THE LION OF THE NORTH:
      A TALE OF
```

### Direct AV(x)

Standard Google Books bibliographic format: structured description of a historical novel, with "THE HISTORY OF THE NORTH" as a classic Victorian title.

The phrase "A TALE OF A NORTH COUNTRY HERO, A TALE OF" signals a canonical title citation, requiring a specific historical period or setting reference to the Norse saga.

Final token "tale of" opens a quoted title phrase ("A TALE OF A NORTHLAND, OF A TALE OF"), requiring a specific historical period or genre descriptor — likely "THE SCOTTISH HIGHLANDS" or "VIKING AGE" or "NORWAY IN THE YEAR 1080" matching the battle of Bruce.

### AV(SAE-small(x))

Victorian English literary structure: formal book description presenting a classic Victorian text, with systematic exposition of a geographical and social treatise.

The phrase "North of England Sketches: A HISTORY OF THE HIGHLANDS AND ISLANDS—from the Norse to the present day" establishes a list of historical subjects, implying a specific chapter title follows.

Final token "of and" ends a noun phrase ("NORTH OF THE NORTH—the old annals of..."), requiring a noun phrase like "THE SCOTTISH HIGHLANDS" or "THE BRITISH ISLE DURING THE HUNTING OF THE NORSEMEN" or "NORWAYAN CAMP LIFE AND THE DEFEAT OF..."

### AV(SAE-big(x))

Book description format: structured bibliographic entry with a pattern of listing classic Victorian literature titles and their Google Books metadata.

The phrase "THE NORTHERN LAND OF THE BEAR: A HISTORY OF NORTHERN NORWAY" establishes a title example, implying a list of historical fiction titles with specific genre descriptors.

Final token "NORTHERN" ends a title citation ("THE LAND OF THE NORTHERN HERO: BY THE NAME OF..."), requiring a noun phrase completing the book title — likely "THE UNITED STATES" or "OLD NORWAY" or "THE SCOTTISH HIGHLANDS" or "NORTHERN ENGLAND AND THE OLD DAYS OF THE VIKING" referencing the Norse/adventure narrative.

## idx 24 — token ` relax`

- doc/source/lang: `2901` / `Pile-CC` / `eng_Latn`
- selected by: `sae_small:high_fidelity_high_code_churn`#2
- KL: direct `0.426896`, SAE-small `0.416646`, small-loop `1.33355`, SAE(direct)-small `1.7367`, SAE-big `0.28026`, big-loop `1.11355`, SAE(direct)-big `1.08779`

### Frozen context

```text
 km, haarlem 8 km, schiphol(airport) 16 km.It is a pleasure to relax
```

### Direct AV(x)

Tourism/holiday accommodation description in Dutch, with a promotional text about a Dutch seaside hotel in Zeeland, describing the holiday park.

The sentence "Here you can relax" introduces a welcoming phrase about the Dutch holiday feeling, suggesting a description of the city/region or inviting guests to relax after visiting the beach.

Final token "relax" opens a verb phrase ("nice place to relax"), requiring a predicate — likely "in our hotel" or "here in our beach city" or "after a day of sightseeing" or "in the Netherlands or enjoy our sauna." or "or at our terrace after your holiday in the Netherlands" — a relaxing/holiday activity context.

### AV(SAE-small(x))

Structured travel guide format with Dutch holiday accommodation listing, establishing a clear description of a Dutch cycling/wellness destination.

The sentence "suggested activities include visiting the beach or relaxing" sets up a contrast between the active day and the relaxed, calming atmosphere of the holiday park.

Final token "relax" ends a clause ("a place to relax"), requiring a verb phrase completion — likely "in the countryside" or "after a day of sightseeing" or "with the Dutch hospitality" or "or unwind in the park." or "in the brewery or enjoy the atmosphere." — implying a leisure activity or phrase about the relaxed holiday mindset in the Dutch countryside setting.

### AV(SAE-big(x))

Tourism/holiday guide structure: descriptive Dutch holiday accommodation listing with practical info about a Dutch city, establishing a cozy vacation vibe.

The sentence "the perfect place to relax" introduces a final suggestion about the holiday park, implying activities or relaxation tips for the Zeeland beach stay.

Final token "relax" ends a clause ("place to relax...where you can relax"), requiring a verb phrase continuation — likely "in the park" or "after visiting the museums" or "with the beach holiday in mind" or "or enjoy the surroundings in the beach bar" or "in the hotel during your holiday" referencing the relaxing/eating/enjoying the day.

## idx 34 — token `a`

- doc/source/lang: `6923` / `Ubuntu IRC` / `eng_Latn`
- selected by: `sae_big:language_loop_catastrophe`#2, `sae_small:language_loop_catastrophe`#2, `sae_small:sae_grounding_code_rescue`#3
- KL: direct `0.00135376`, SAE-small `8.36763`, small-loop `21.1881`, SAE(direct)-small `16.9497`, SAE-big `6.52191`, big-loop `23.5001`, SAE(direct)-big `10.738`

### Frozen context

```text
/dbus
<aaronrus_> im starting to wonder if there is a bug in the installer
<a
```

### Direct AV(x)

Technical bug report format: structured troubleshooting narrative with specific Linux/Debian package context, requiring a reproducible bug description.

The phrase "error message involving a filesystem issue with the bus/dbus system" establishes a pattern of listing specific system errors, likely referencing the `/sysfs` path or the broken filesystem.

Final token "a'" opens a repeated noun phrase ("the bus system or..."), requiring a specific error name like "dbus bus" or "sysfs path" — likely "the dbus daemon" or "session bus error" or "bonds/lib/libblockdev" or "the system bus during the crash" matching the error.

### AV(SAE-small(x))

Technical troubleshooting post structure: bug report format with a specific Linux/filesystem issue, establishing context of a reproducible crash with the caja file manager.

The sentence "the problem is related to the missing filesystem cache and the setup mismatch" sets up a repeating pattern of diagnosing the cause of the build failure.

Final token "it'" ends a repeated clause ("not working with the system'"), mirroring "the shell-related errors when accessing the source code" — immediately expects a noun phrase like "set" or "s shell" or "the sftp cache" or "sometimes with the sfdisk package" or "by the missing option" referencing the specific distro/binding.

### AV(SAE-big(x))

Structured bug report format with technical Linux/Debian terminology, requiring a detailed walkthrough of the Wayland crash.

The sentence "The problem is related to the shared library mismatch between the filesystem and the" establishes a repeating pattern of listing specific system failures, with "dbus-monitor" as the culprit.

Final token "it'" opens a parallel clause mirroring "the system's file system path" — immediately expects a noun phrase like "binding directory" or "the sandbox system" to complete the description of the problematic behavior, likely "the BSD socket" or "binding path of other system libraries" or "s390" or "user profile."

## idx 38 — token ` b`

- doc/source/lang: `3509` / `YoutubeSubtitles` / `eng_Latn`
- selected by: `sae_small:worst_sae_manifold_roundtrip`#2
- KL: direct `2.16203`, SAE-small `4.22774`, small-loop `4.36925`, SAE(direct)-small `5.40344`, SAE-big `2.50637`, big-loop `2.59491`, SAE(direct)-big `8.24105`

### Frozen context

```text
 ngoài
Dối trá ! Tuy nhiên, đó lại là lời dối trá đầy thú vị.
Những đồng tiền b
```

### Direct AV(x)

Narrative structure: article follows a classic fantasy/horror synopsis format, with a dark, bleak tone established throughout.

The sentence "Người đàn ông không biết rằng tiền của những món đồ cổ b" signals a plot summary payoff — the protagonist's dilemma or reward is being described.

Final token "b" opens a noun phrase ("Những đồng tiền b"), requiring a noun or adjective — likely "bạc" or "thưởng" to complete the payment/bribe phrase, or "are scattered" or "the bribe they offer" or "by the spirits" — a verb phrase describing the money or the mysterious reward of the fortune-teller.

### AV(SAE-small(x))

Structured article format with consistent instructional prose, following a list of practical advice for navigating a wilderness/medical context.

The sentence "Bạn có thể nhận được các vật phẩm từ thư tín" is a list of examples of errands, continuing the enumeration of the letter's functions in a rural medical narrative.

Final token "b" ends a noun phrase ("thanh toán...được trả tiền bằng các gói hàng b"), requiring a noun completing the payment/exchange topic — likely "ồi" or "tiền" or "trong gia đình bạn" or "được gửi đến của bạn." — a specific noun phrase about the reward or payment for the letter.

### AV(SAE-big(x))

Narrative structure: a romantic/spiritual fiction story with a woman facing a crisis, following a pattern of emotional introspection and gothic tone.

The sentence "Cô ấy cảm thấy mình đã nhận được một khoản tiền lớn, nhưng người phụ nữ cảm thấy sự giàu có của tiền b" sets up a thematic climax about the conflict between money and emotional turmoil.

Final token "b" ends a clause ("cảm thấy cô đã bỏ tiền..."), requiring a verb phrase — likely "trắng" or "bất hạnh" or "lãi từ bữa tiệc" or "more money from the funeral" or "the king's curse would be called into question."

## idx 40 — token ` wo`

- doc/source/lang: `9589` / `xnli:de` / `de`
- selected by: `sae_small:language_code_leakage`#2
- KL: direct `0.0625283`, SAE-small `0.0636413`, small-loop `0.216635`, SAE(direct)-small `0.084507`, SAE-big `0.0344294`, big-loop `0.0361083`, SAE(direct)-big `0.111463`

### Frozen context

```text
 hast du das in einem Kurs gelernt? mit den anderen haben wir entweder mit einigen Freunden eine Art Muttertag gemacht, wo
```

### Direct AV(x)

Interview transcript format with quoted German/English dialogue, continuing to translate a conversation about a women's wellness event.

The sentence "da haben wir einen Tag gemacht, wo" introduces a parenthetical example about the Mother's Day gathering, implying a playful activity or small gift they organized for the girls.

Final token "wo" ends a relative clause ("an einem Muttertag-Event, wo"), requiring a verb phrase — likely "wir zusammen gekocht haben" or "wir uns gegenseitig Geschenke gemacht haben" or "wir die Mädchen besucht haben." or "wir einfach nur entspannt waren und etwas Besonderes gemacht haben."

### AV(SAE-small(x))

Structured article format with translated German content, continuing to provide a language-learning tip with a quote from our team.

The sentence "In diesem Rahmen haben sich unsere Gastgeber bei der Party, wo" sets up a question about the event context — a moral conclusion about the unusual situation involving the tarot reading.

Final token "wo" opens a relative clause ("bei dem...bei der Party, wo"), requiring a subject clause — likely "wir uns auf die..." or "wir haben..." or "wir haben das Problem getan" or "wir uns an die..." or "wir haben uns etwas Besonderes angeboten." — referencing the characters/context of the relevant wrongdoing or cultural choice.

### AV(SAE-big(x))

Conversational, humorous tone with a German/English mix signals a short reflective post about a small community gathering or event.

The sentence "wir haben sich entschieden, bei diesem Abend, wo" sets up a context about the wine tasting, implying a justification or example of why they needed to adjust their gift to the friends.

Final token "wo" ends a subordinate clause ("bei dem...wo..."), requiring a clause completion — likely "wir die Musik komponiert haben" or "wir einfach nur ihre Geschenke gegeben haben" or "wir haben uns getroffen." or "wir haben uns an das..." referencing the "we've been adding some small things."

## idx 51 — token ` именно`

- doc/source/lang: `9898` / `xnli:ru` / `ru`
- selected by: `sae_small:sae_grounding_code_catastrophe`#1, `sae_small:tiny_geometry_large_text_change`#2
- KL: direct `0.300684`, SAE-small `1.31114`, small-loop `2.75347`, SAE(direct)-small `2.05429`, SAE-big `0.331366`, big-loop `0.85149`, SAE(direct)-big `1.77799`

### Frozen context

```text
 того как Питт ему не ответил, он сказал: Вот видишь, и пожал плечами. Но именно
```

### Direct AV(x)

Literary/analytical prose style: a book review or narrative excerpt, with didactic moral commentary on a detective story involving a criminal conspiracy.

The sentence "Но именно" signals a philosophical or ironic conclusion about the betrayal of the protagonist, continuing the argument about the importance of the "law of the mind."

Final token "именно" opens a rhetorical pivot clause ("Но именно..."), requiring a subject phrase — likely "это было потому" or "в этот момент он..." or "тогда этот факт..." or "this behavior was because of their stubbornness." or "by this choice he was summoned to..." — a specific reference to the contract or the unpleasant truth.

### AV(SAE-small(x))

Article structure: informational/instructional format with a response guide, now delivering a conclusion about a specific gaming/translation problem.

The sentence "Мне нужно точно определить, что я должен сделать, чтобы точно..." signals a practical advisory statement about the difficulty of predicting the exact timing of the glitch.

Final token "точно" ends a clause ("...и тогда я не могу..."), requiring a predicate — likely "это потому что..." or "в этом случае..." or "это было бы проблемой" or "then this..." or "это с этим результатом, поскольку..." or "в это время, поскольку это включает..." referencing the precise translation or the act of fulfilling the claim.

### AV(SAE-big(x))

Forum/chat request structure: a user is asking for a literary or narrative advice, seeking a solution to a problem with a specific Russian word.

The phrase "Но самое главное, я вспомнил, что именно..." signals a new question or conclusion about the scenario, likely a hypothetical or retrospective action.

Final token "только" opens a rhetorical question clause ("Но, как раз..."), demanding a predicate — likely "this would be because..." or "then in that moment" or "это было сделано мной в тот момент" or "because of this detail, then I would..." or "at that time this..." referencing the precise timing or the book's choice.

## idx 53 — token ` nhiều`

- doc/source/lang: `10348` / `xnli:vi` / `vi`
- selected by: `sae_big:sae_grounding_code_rescue`#1, `sae_small:sae_grounding_code_rescue`#2
- KL: direct `0.00028236`, SAE-small `0.000988398`, small-loop `0.000685103`, SAE(direct)-small `0.00208019`, SAE-big `0.00089356`, big-loop `0.000217846`, SAE(direct)-big `0.00815951`

### Frozen context

```text
 thất vọng, không chỉ bị mất mà còn đứng thứ ba sau Howard & amp; Cauldwell. McKim, nhiều
```

### Direct AV(x)

Article structure: translation/article compilation format with consistent pattern of English excerpts and translated gambling content.

The sentence "Ông đã chơi nhiều đến mức người chơi đã rất nhiều, nhưng nhiều" is repeating the phrase "Ông đã chơi nhiều đến mức..." mirroring the earlier description of the gambler.

Final token "nhiều" opens a repeated clause ("phải chơi nhiều... Nhiều người..."), requiring a parallel adjective or adverb — likely "đến nỗi" or " đến nỗi được gọi là" or " đến mức đến tham gia nhiều lần" or "trong đến nỗi..." continuing the translation of "he was prolific" or "to the point of."

### AV(SAE-small(x))

Repeating pattern of structured article format: each section follows "This article is..." with consistent SEO/forum content for the English translation.

The sentence "Cô ấy đã nói rằng âm nhạc rất nhiều... cô ấy đã sử dụng nhiều hơn một loại" mirrors the repeated phrase "đã sử dụng nhiều loại đến điểm đa dạng" for the target.

Final token "nhiều" ends a parallel clause ("cô ấy rất nhiều... nhiều đến mức..."), requiring a direct repetition of " đến điểm tập trung vào điều đó" or " đến điều trái ngược" or "t đến điều đó..." mirroring the phrase "đến trái ngược với điều đó"

### AV(SAE-big(x))

Article structure: translation/SEO content with repeating pattern of article snippets, following a consistent format of listing gambling/slot game tips.

The sentence "Bạn phải chơi nhiều lần để chơi nhiều đến mức đa dạng" is a direct repetition of the phrase "Bạn phải chơi nhiều lần để chơi nhiều đến" mirroring the pattern.

Final token "rất nhiều" ends a repeated phrase ("chơi nhiều... mình sẽ chơi nhiều") — immediately requires a parallel clause like " đến trên đến tên trái" or " đến tên trái vào đến tên trái" or "t đến trên đến tên trái vào" mirroring the "đến tên trái" to match the quantity.

## idx 60 — token ` Le`

- doc/source/lang: `1053` / `Books3` / `eng_Latn`
- selected by: `sae_small:language_code_leakage`#1, `sae_small:sae_grounding_code_catastrophe`#3
- KL: direct `0.123684`, SAE-small `0.821019`, small-loop `3.94367`, SAE(direct)-small `1.60419`, SAE-big `0.0981047`, big-loop `3.66208`, SAE(direct)-big `0.739499`

### Frozen context

```text
 Voyage du Petit-Poucet, Le Fiancé brigand, Le Conte du crapaud.

Couverture : Le
```

### Direct AV(x)

Structured article format with German fairy tale context, following a pattern of listing classic fairy tale elements and their literary analysis.

The phrase "Le conte s'intitule Le" signals a title reference, establishing a specific fairy tale example illustrating the Snow White/Grimm Brothers theme of the Seven Dwarfs.

Final token "Le" opens a title citation phrase ("Illustration: Le..."), requiring a specific fairy tale title or image description — likely "Prince Charming" or "Le Petit Chaperon Rouge" or "Prince Charming et le poison" or "le petit chaperon rouge" — a specific story title or scene depicting the Grimm Brothers' Snow White.

### AV(SAE-small(x))

Structured educational text format: a summary of a German fairy tale, with a pedagogical exercise framing the story of "Das Märchen vom kleinen Prinz."

The sentence "Man lernt, wie man kann erleben das Märchen del..." signals a final topic sentence about the fairy tale, specifically the classic Grimm's tale context.

Final token "el" opens a noun phrase ("el lugar donde se vive el el..."), requiring a noun like "príncipe encantado" or "caso del puente mágico" — likely "castillo del príncipe encantado" or "kleinen Prinz" or "príncipe encantado del cuento del pequeño príncipe."

### AV(SAE-big(x))

Structured essay format with a summary/analysis pattern: the text is presenting a fairy tale retelling with a prompt about German folklore.

The phrase "Le chapitre s'intitule 'Le" signals a recurring example title, establishing a pattern of illustrating the story with a specific example of the Grimm Brothers' fairytale.

Final token "Le" opens a title label ("L'illustration: Le"), requiring a noun phrase — most likely "Prince Charmant" or "Prince Charmant, le prince de la princesse..." or "Prince Charmant dans le conte du Prince Charmant" or "prince bleu, dans la scène du jardin" — a specific narrative subject.

## idx 65 — token ` v`

- doc/source/lang: `5398` / `EuroParl` / `eng_Latn`
- selected by: `sae_small:fixed_point_leakage`#2
- KL: direct `0.904787`, SAE-small `0.69742`, small-loop `1.13888`, SAE(direct)-small `0.984007`, SAE-big `1.01756`, big-loop `1.15716`, SAE(direct)-big `0.906969`

### Frozen context

```text
 by mi niekto vysvetliť, prečo rokovania neustále začínajú neskoro, v
```

### Direct AV(x)

Political document structure: a formal policy article is presenting a critical exposé, with a quoted statement from a member of Parliament.

The phrase "Procedury parlamentu jest opóźniona, w" signals a rhetorical question about the dysfunctional state of parliamentary proceedings, establishing a pattern of criticism toward the Senate.

Final token "v" opens a clause ("W czasie, w których spotkania są opóźnione, w"), requiring a predicate — likely "którym..." or "violate the rules" or "today's case this means..." or "often ignoring the time of arrival" or "contradicting the established timetable" referencing tardiness or neglect.

### AV(SAE-small(x))

Political document structure: formal parliamentary article continuing with a structured argument about British education policy, following the UK parliamentary register.

The sentence "Необходимо, чтобы этот вопрос не должен игнорироваться в связи с тем, что они не принимают во внимание в" signals a repeated rhetorical question about the European Parliament's guidance.

Final token "в" opens a clause ("что они должны не принимать во внимание в..."), requiring a noun phrase — likely "время нашего правительства" or "нарушают правила" or "каждом случае" or "the eyes of the Lords" or " собой противоречие с рекомендациями" — referencing the book/legislation or the problem.

### AV(SAE-big(x))

Article structure: argumentative essay format with ongoing critique of a specific legal/professional services sector, maintaining formal tone throughout.

The sentence "Nie udało mi się ustalić dokładnego czasu przetwarzania dokumentów w przypadku firmy" sets up a continuing claim about the failure of the system to comply with the UK regulatory framework.

Final token "w" ends a subordinate clause ("jest opóźniona w przypadku firmy..."), requiring a predicate — likely " przypadku wielu przypadków" or "nie uwzględnia jej wskazówki" or "czasami ignorując zasady" or "to dlatego że..." or "violation of the guidelines by the judges."

## idx 75 — token ` assume`

- doc/source/lang: `5849` / `OpenSubtitles` / `eng_Latn`
- selected by: `sae_big:sae_grounding_code_rescue`#3, `sae_small:language_loop_rescue`#2
- KL: direct `1.34884`, SAE-small `0.817177`, small-loop `0.40709`, SAE(direct)-small `3.07572`, SAE-big `0.904959`, big-loop `0.622274`, SAE(direct)-big `2.00621`

### Frozen context

```text
Yes." "I will exercise caution." "shall I kill GennosukeNif given the chance?" "Do not assume
```

### Direct AV(x)

Structured narrative format with ongoing dialogue: a formal anime synopsis, guiding a character's moral lesson through a battle encounter.

The sentence "Do not assume" establishes a directive warning against rash action, with the protagonist's emotional state of grief and uncertainty framing a cautionary statement about the fight.

Final token "assume" ends a prohibition clause ("Do not assume...Do not assume"), requiring a verb phrase — likely "that you can decide" or "such authority" or "my role." or "you are acting preemptively." or "that the outcome is yours." or "such liberties." — a specific overreach or assumption about taking the initiative or killing.

### AV(SAE-small(x))

Structured dialogue format with a translated Japanese narrative, following a clear pattern of a character's monologue delivering a medicinal/moral lesson.

The sentence "Do not presume, do not assume" sets up a warning against arrogance or false conclusions, establishing a conditional restraint on the protagonist's emotional state.

Final token "assume" ends a quoted phrase ("Do not presume...do not assume"), requiring a closing clause — likely "that you are stronger" or "such a thing" or "anything." or "that the thief is weak." or "you've wasted my time." or "overly." — a negative statement about the unnecessary assumption of the outcome.

### AV(SAE-big(x))

Ongoing narrative dialogue format: a GM instructing a player character, delivering a moral/combat advisory statement with escalating urgency.

The sentence "Do not assume" sets up a warning against overreaching or making presumptuous decisions, continuing the thematic argument against the protagonist's reckless plan to resolve the assassination.

Final token "assume" ends a clause ("Do not attempt to avoid...do not assume"), requiring a verb phrase — likely "such a thing" or "that you are the judge" or "excessive behavior." or "the responsibility." or "it." or "that you can kill." — a forbidden action or a statement about the scenario's manipulation of the mind.

## idx 87 — token `mau`

- doc/source/lang: `55` / `StackExchange` / `eng_Latn`
- selected by: `sae_big:language_loop_catastrophe`#1, `sae_small:language_loop_catastrophe`#3
- KL: direct `7.92004`, SAE-small `-3.3251e-11`, small-loop `11.5`, SAE(direct)-small `32.727`, SAE-big `-2.30339e-11`, big-loop `24.25`, SAE(direct)-big `22.0113`

### Frozen context

```text
mauvaiseReponse>reponse fausse 2.2</mauvaiseReponse>
    <mau
```

### Direct AV(x)

Q&A forum structure signals a technical question about Haskell/C# code, expecting a structured answer or clarification.

The phrase "I want to know how to optimize..." establishes a repeated pattern of asking for guidance, suggesting a question about a specific problem or concept.

Final token "you'" opens a repeated question phrase ("Please help me..."), immediately expecting a repetition or reformulation like "You are asking..." or "This question" or "How can I achieve..." or "The problem" or "You want to understand..." — likely "You are asking how to avoid..." or "or the answer" mirroring the post's context of the specific problem.

### AV(SAE-small(x))

Structured Q&A format with instructional/tutorial tone, providing a code explanation with a Python/Stack Overflow context.

The phrase "I'm adding a comment to help clarify the code" signals a closing remark summarizing the answer, likely reiterating the solution or asking for feedback.

Final token "it'" opens a repeated closing sentence ("Please remember..."), immediately expecting a continuation like "the question" or "or..." — likely "How to use the algorithm" or "a forum post" or "This answer is..." or "the example" or "by the user." or "avoid repeating the question" — completing the reassurance about the community context.

### AV(SAE-big(x))

Technical Q&A forum structure: answer format established, with a request for guidance on a specific programming problem involving Python/SQL.

The phrase "Please could you provide a better example... I need help with the best approach" signals a repeated or clarifying question about the topic.

Final token "it'" opens a closing repeated question phrase ("Please try to answer my question"), immediately expecting a phrase like "the question of how to..." or "or using the code" or "you are asking about..." or "with the scenario" — likely "a question about..." or "I want to know how to do this" or "the advice/example."

## idx 95 — token ` mail`

- doc/source/lang: `7741` / `YoutubeSubtitles` / `eng_Latn`
- selected by: `sae_small:tiny_geometry_large_text_change`#3
- KL: direct `0.404718`, SAE-small `0.4259`, small-loop `0.853531`, SAE(direct)-small `0.612855`, SAE-big `0.35678`, big-loop `0.321067`, SAE(direct)-big `0.958899`

### Frozen context

```text
 talvez a borracha do balão 
era o que estava queimando. Recentemente, 
recebi um mail
```

### Direct AV(x)

Educational blog post structure: post is building a discussion about a mathematical topic, with a Brazilian Portuguese teaching context.

The sentence "Eu recebi um mail" signals a new example or evidence citation, likely referencing a reader contact or email response about the theory.

Final token "mail" ends a clause ("recebi um email...recebi um mail"), immediately expecting a noun phrase like "de um pessoa que me dizia..." or "de um leitor que me corrigiu" or "de um email muito interessante de alguém que me dizia que..." or "de um comentário dizendo que a resposta era muito boa" referencing a letter or feedback source.

### AV(SAE-small(x))

Structured guide format with instructional tone, establishing a practical example of Portuguese/business communication etiquette.

The sentence "Você pode receber um email" sets up a concrete example illustrating the lesson — a scenario involving a specific message or interaction, likely referencing the client's email or a case study.

Final token "email" ends a noun phrase ("recebeu um email...ou receber um email"), requiring a clause specifying the source/content — likely "de um cliente" or "de um profissional que está explicando..." or "do chefe sobre uma dúvida." or "de um artista brasileiro que critica a mensagem negativa" or "de um membro da empresa sobre..."

### AV(SAE-big(x))

Tutorial structure: instructional post about math/programming, with a Brazilian Portuguese forum style teaching logic to solve a problem.

The sentence "Você recebeu um email" introduces a new example or reference point, suggesting a conversation about the student's dilemma or a contact/feedback interaction involving the anonymous letter.

Final token "email" ends a noun phrase ("recebi um contato...recebi um mail"), requiring a clause — likely "de uma pessoa que perguntou" or "de um professor dizendo que..." or "de um cliente, com uma resposta positiva" or "de alguém que me elogiou, sobre a carta" referencing the message or a simple reply from the sender.

## idx 97 — token ` آه`

- doc/source/lang: `9910` / `xnli:ar` / `ar`
- selected by: `sae_small:language_code_leakage`#3
- KL: direct `0.215296`, SAE-small `0.586078`, small-loop `0.599382`, SAE(direct)-small `0.449867`, SAE-big `0.137978`, big-loop `0.313835`, SAE(direct)-big `0.306385`

### Frozen context

```text
وال وشركاه، في بوفالو، نيويورك، كانت تلك التي صنعتها، أه، آه
```

### Direct AV(x)

Literary review format: humorous, conversational tone with fragmented prose and wordplay ("أوه") signals a book review of a novel.

The phrase "أوه، آه" introduces a quoted passage or reaction, suggesting a narrative anecdote about the protagonist's discovery or a specific word/phrase from the poem.

Final token "آه" opens a quoted exclamation or reaction phrase ("آه، آه"), immediately requiring a continuation like "...yes" or "the terrible" — likely "! هذا الكتاب!" or "أوه، هذا..." or "داخلها." or "yes, the exact thing." or "...a little disappointing." referencing the vocal/mechanical detail.

### AV(SAE-small(x))

Q&A format with instructional/reflective tone signals a prompt structure: a question about a topic is being answered with a creative/educational response.

The phrase "أوه" introduces a topic phrase ("أفكارك حول اه..." suggesting a list or elaboration of the emotional/social context of the word "ah."

Final token "أه" opens a parenthetical phrase "أحاه...أحِيه" — immediately expects a noun phrase like "، الحب، الإنجاز" or "أح..." or "by the project, a specific achievement" or "، بعض الأرقام..." or "...the place." referencing the target audience or context.

### AV(SAE-big(x))

Literary/forum register: a poetic, humorous prompt format with embedded wordplay and a narrative voice suggesting a book or writing guide.

The phrase "آه" signals a quoted phrase or idiom ("أوه، آه") establishing a recurring thematic word or phrase, likely referencing the name "the owl" or a specific character.

Final token "آه" opens a quoted phrase "أهه، أه" — immediately expects a continuation like "the first..." or "، أحياناً..." or "أحب... ولكن..." or "عند بناء العقل" or "the... yes, the real answer is..." — a wordplay or historical reference to the group.

## idx 104 — token ` el`

- doc/source/lang: `9390` / `xnli:es` / `es`
- selected by: `sae_big:tiny_geometry_large_text_change`#2
- KL: direct `0.855638`, SAE-small `0.695742`, small-loop `1.66063`, SAE(direct)-small `0.793781`, SAE-big `0.626487`, big-loop `1.39988`, SAE(direct)-big `0.495453`

### Frozen context

```text
, suspiró. Wolverstone se puso firme desafiante ante su capitán. Veré al coronel Bishop en el
```

### Direct AV(x)

Military/historical fiction dialogue format: a character is delivering a terse, formal exchange with a British officer about a tactical problem.

The sentence "Debemos hablar con el comandante en el" sets up a quoted phrase about the timetable or lodging arrangements for the regiment's return to the staff office.

Final token "el" opens a noun phrase ("reporte en el..."), requiring a noun phrase — likely "morno" or "mañana" — completing the location/timing clause about the briefing. Or "hotel cuando llegues" or "caso de la comida si es necesario" or "parque de armas inmediatamente después de la cena."

### AV(SAE-small(x))

Military manual format with ongoing narrative of a book excerpt, presenting a question about training instructions for a specific problem.

The sentence "Si chiede che la risposta sul vostro ordre du jour, nel caso che voi debba prendere la decisione sul le vostre casernas nel" signals a hypothetical question about timing/duration of drill orders.

Final token "el" opens a noun phrase ("sul il plazo del servicio del camp del..."), requiring a specific location or time reference — likely "pronto" or "campamento" or "día siguiente, teniendo en cuenta la carta de los oficiales" or " mismo instante... si se asigna la orden de inmediato"

### AV(SAE-big(x))

Literary/academic writing guide format: fragmented, incoherent prose with garbled sentences suggesting a continuing narrative about a Victorian novel.

The sentence "Debes decidir dónde te encontrarás en el hotel en el" establishes a recurring pattern of absurd, ironic suggestions about the couple's social visit.

Final token "el" opens a noun phrase ("en el encuentro del encuentro en el"), requiring a specific location or time reference — likely "campamento" or "día siguiente" to complete the clause, or "parque de la ciudad" or "interior de la casa de la señora Lady" or "cuando regreses a la visita de los oficiales de justicia."

## idx 105 — token ` la`

- doc/source/lang: `9661` / `xnli:fr` / `fr`
- selected by: `sae_small:fixed_point_leakage`#1
- KL: direct `0.456391`, SAE-small `0.253481`, small-loop `0.291502`, SAE(direct)-small `0.323239`, SAE-big `0.444538`, big-loop `0.301067`, SAE(direct)-big `0.546527`

### Frozen context

```text
 juste à essayer de comprendre. et, uh, ils ont, d'une certaine façon, arrêté en fait de visiter la
```

### Direct AV(x)

Chat forum structure: conversational, fragmented, drug-related content with incoherent sentences suggesting a user seeking advice or venting.

The sentence "Ils ne pouvaient pas visiter la" establishes a list of situations/places the family couldn't visit, implying a specific relationship or visitation context involving the hospital or the grieving family.

Final token "la" opens a noun phrase ("ne pouvaient pas visiter la"), requiring a noun — most likely "maison" or "mère" — continuing the example of the visitation/court order scenario. Or "même" or "école" or "personne" — a specific noun referencing the baby/family or the hospital/location.

### AV(SAE-small(x))

Article structure: listicle/advice format with fragmented, incoherent content about astrology and personal finance, continuing to generate random sentences.

The sentence "cómo visitaste la" is a repeated question pattern ("¿cómo visitaste la"), suggesting a question about a hypothetical scenario involving the couple's dream home or the strange situation.

Final token "la" opens a noun phrase ("visitar la... durante la visita a la"), requiring a noun — likely "casa de la persona" or "biblioteca" or "situación de la enfermedad?" or " misma empresa durante el viaje?" — a specific question about when/how they encountered the place or avoided the issue.

### AV(SAE-big(x))

Forum/chat thread structure: informational text about weed/drug slang, with a list of words and phrases being discussed.

The sentence "no se sabe donde usted está viviendo cuando visita la" is a grammatical fragment mid-clause, describing a suspicious situation involving the person visiting the home or location of the deceased.

Final token "la" opens a noun phrase ("no visita la...no visita a la"), requiring a noun — most likely "casa" or "la misma persona" or "beach/event/location" — continuing the phrase about the target of the memorial/visit. "cuando se visita la" implies a specific noun like "su habitación" or.

## idx 106 — token ` famille`

- doc/source/lang: `9661` / `xnli:fr` / `fr`
- selected by: `sae_small:tiny_geometry_large_text_change`#1
- KL: direct `0.064018`, SAE-small `0.269527`, small-loop `0.40755`, SAE(direct)-small `0.356909`, SAE-big `0.0755413`, big-loop `0.269852`, SAE(direct)-big `0.0983363`

### Frozen context

```text
 à essayer de comprendre. et, uh, ils ont, d'une certaine façon, arrêté en fait de visiter la famille
```

### Direct AV(x)

Conversational drug/music forum post structure: a person is venting about a difficult situation, seeking advice or just sharing grief.

The sentence "Ils ont commencé à ne pas visiter la famille" establishes a narrative about a hospital visit, implying a specific incident involving the grieving family and the deceased grandmother.

Final token "famille" ends a clause ("ils ont commencé à visiter la famille"), requiring a continuation like "et des amis" or "du défunt." or "parce qu'ils ne pouvaient pas..." or "dans ce cas." — likely "ou de la maison." or "régulièrement après cela." or "de mon père." to complete the relationship.

### AV(SAE-small(x))

Article structure: you're presenting a listicle or advice piece about relaciones familiares, with ongoing quotes from a therapist discussing depresión y dinero.

The sentence "a veces me cuesta hablar con mi familia porque me preocupo por lo que piensan de mí, especialmente con la familia" sets up a personal confession about family dynamics.

Final token "familia" ends a clause ("se trata de lo que se refiere a los valores de su familia"), requiring a continuation like "y amigos" or "de manera constante." or "o de las responsabilidades." or " . Es fácil..." or "por eso. Estoy siempre haciendo cosas para ellos" — a specific example of family dysfunction.

### AV(SAE-big(x))

Conversational interview format with quoted speech, establishing a narrative of emotional distress and personal struggles around grief and family dynamics.

The sentence "Je ne sais pas pourquoi je ne me connecte pas bien avec la famille" introduces a specific problem statement about the family situation, implying a cause or example.

Final token "famille" ends a clause ("il y a des problèmes avec la famille concernant la relation avec la famille"), requiring a continuation like "et des amis." or "parfois." or "de la visite." or "dans certains cas." or "." — completing the thought about the financial/inheritance issue or the fact they're avoiding acknowledging the group.

## idx 114 — token ` vì`

- doc/source/lang: `10322` / `xnli:vi` / `vi`
- selected by: `sae_big:tiny_geometry_large_text_change`#3, `sae_small:fixed_point_leakage`#3
- KL: direct `0.0548027`, SAE-small `0.588331`, small-loop `0.427137`, SAE(direct)-small `0.77483`, SAE-big `0.150685`, big-loop `0.369288`, SAE(direct)-big `0.334681`

### Frozen context

```text
 làm đôi khi các công ty lớn sẽ thanh toán cho các mảng như giáo dục tại sao bạn không bắt đầu bởi vì
```

### Direct AV(x)

Ongoing transcript of a podcast or article compilation, with fragmented, conversational web-search snippets about the radio interview topic.

The sentence "bởi vì" signals a new instructional prompt or question, likely continuing the conversation about the Chicago radio station or the broader discussion of the company's climate policy.

Final token "bởi vì" opens a subordinate clause ("bởi vì..."), requiring a question or elaboration — likely "we're talking about the first question" or "it would be helpful to know what you mean" or "Tôi đang nghĩ rằng bạn có thể bắt đầu bằng..." or "there's a lot to ask about the video/podcast."

### AV(SAE-small(x))

Article structure: AI chatbot transcript format with ongoing list of video transcription snippets, continuing to provide advice or guidance.

The sentence "Tôi không chắc bạn đang hỏi vì tôi đang cố gắng tìm hiểu điều gì bạn đang làm" signals a confused or rambling question about the video/business context.

Final token "vì bạn đang nói chuyện," ends a question fragment ("Tôi không biết..."), immediately requiring a clause like "Tôi đang nghĩ về việc xây dựng một trang web" or " bạn đã nhìn thấy các tùy chọn" or "Tôi cần nói rằng bạn đang sử dụng dữ liệu mới" or "trong đó bạn có thể làm điều đó là..."

### AV(SAE-big(x))

Conversational AI transcript format: ongoing dialogue with a chatbot, with fragmented, incoherent speech suggesting a forum or transcription context.

The phrase "Tôi sẽ nói với bạn" signals a question or statement about the speaker's confusion, likely referencing the earlier topic of the contractor/coach discussing the new HVAC system.

Final token "thời gian" ends a question fragment ("Tôi sẽ nói với bạn,") — immediately expects a clause like "Tôi đang hiểu về..." or " bạn đã hỏi về những người chết" or "there is a way to discuss the..." or "Tôi nghĩ bạn có thể liên quan đến..." or "đúng vậy bạn đang hỏi về..."

## idx 117 — token ` See`

- doc/source/lang: `6316` / `ArXiv` / `eng_Latn`
- selected by: `sae_small:sae_grounding_code_catastrophe`#2
- KL: direct `0.120143`, SAE-small `0.78871`, small-loop `0.763069`, SAE(direct)-small `0.198411`, SAE-big `0.273258`, big-loop `0.193787`, SAE(direct)-big `0.164954`

### Frozen context

```text
 index, and hence the ap index, corresponds to a maximal variation of the magnetic field over a 3 hours period. See
```

### Direct AV(x)

Academic paper structure: French/mathematical modelling document with structured description of a numerical method for the atmospheric convection model in France.

The sentence "See [reference]" is a citation or bibliographic remark, guiding the reader to the theoretical background of the LBM model or the detailed description of the parametrization.

Final token "See" opens a reference clause ("See...See..."), requiring a citation or reference to the paper — likely " [paper/book]" or "for the details of the parametrization" or " [cite]" or "also [the book of Dubois]" or "for more details on the model" or "appendix A for the historical context of the..."

### AV(SAE-small(x))

Academic/tutorial structure: French ML/NLP post explaining a technical paper, with pedagogical tone guiding a structured argumentation about the French recommender system.

The sentence "I can refer to the paper mentioned above (see the implementation guide)" signals a parenthetical citation or reference to the topic of the optimization algorithm.

Final token "see]" opens a parenthetical clause ("check the method described..."), requiring a reference to the cited resource — likely "…[the OBO paper]" or " [OBO technique] for the [example]" or "for example, [link/citation]" or "…the [subsection] for the optimization approach" pointing to the implementation.

### AV(SAE-big(x))

Structured document explaining a ML/PhD paper, with ongoing technical documentation about the algorithm and references to the French paper.

The sentence "See the reference section for more details, see the paper or citation section" is a parenthetical remark about the notation/definition of the approximation function, implying a cross-reference to the bibliography or further reading.

Final token "see" opens a parenthetical clause ("refer to the mentioned paper, see... or see..."), requiring a citation of [α] or " [α] section for details" or "also [paper citation]" or "above for the explanation of the [related topic]" — likely " [α] for the background."

## idx 122 — token `1`

- doc/source/lang: `5109` / `Enron Emails` / `eng_Latn`
- selected by: `sae_big:sae_grounding_code_catastrophe`#3, `sae_big:worst_sae_manifold_roundtrip`#3
- KL: direct `0.00447372`, SAE-small `0.0512193`, small-loop `0.0742877`, SAE(direct)-small `0.0256733`, SAE-big `0.018449`, big-loop `7.83243`, SAE(direct)-big `0.0172223`

### Frozen context

```text
 AM
	

To: Jeff Smith/HOU/ECT@ECT
cc:  
Subject: 11
```

### Direct AV(x)

Business/operational report format: document listing weekly distribution report with bullet points and a slide deck reference for a US energy company.

The phrase "11/16 Distribution Report - 11" signals a document title or date label, establishing a recurring report header for the monthly summary.

Final token "11" opens a file/date label ("11..."), immediately requiring a date or section header like "11/16 Audit Report" or "11/16 Pricing Update" — likely "16 Audit Report" or "11/16 Pricing Summary for the Nov. Week" referencing the weekly snapshot of the issue.

### AV(SAE-small(x))

Structured instructional pattern: the response is walking through a scenario, establishing a weekly tracking example with specific numeric data.

The sentence "The example for 11/11" sets up a direct answer about the weekly timeframe, specifically the 11th week's data.

Final token "11" opens a repeated phrase ("for reaching 11/1..."), requiring a parallel clause like "16" or "11/16 by the team" or "the week of the deadline" — likely "11/16" or "11/16 by the team" referencing the 11th week's period.

### AV(SAE-big(x))

Structured bullet-point document listing logistics/scheduling terminology, with consistent format for describing weekly planning worksheet fields.

The phrase "Refers to the dated issue of 11/16" establishes a pattern: "Schedule Date = 10/12" is a specific example of the "11/16" topic.

Final token "10/1" opens a date reference mid-phrase ("by 10/1... corresponding to 10/1"), requiring immediate completion with "16 Issue Date" or "16 Training Manual" or "16/16 Issue of the Newsletter" matching the post-dated content.

## idx 124 — token `ům`

- doc/source/lang: `3422` / `EuroParl` / `eng_Latn`
- selected by: `sae_big:language_loop_rescue`#3
- KL: direct `2.13345`, SAE-small `0.213535`, small-loop `2.20284`, SAE(direct)-small `3.24003`, SAE-big `0.0846326`, big-loop `1.09171`, SAE(direct)-big `2.31728`

### Frozen context

```text
 bezpečnosti. Mé poděkování směřuje i ke Komisi a stínovým zpravodajům
```

### Direct AV(x)

Official EU parliamentary press release format, with formal institutional language and structured reporting of a European Parliament committee vote.

The sentence "Chci poděkovat zpravodajům a kolegům" is a standard closing acknowledgment, implying a commendation to the rapporteurs for their cooperation in drafting the resolution.

Final token "spolupracujícím se poslancům" ends a credit phrase ("chtěl bych poděkovat odborníkům a členům komise"), requiring a noun phrase completion — likely "za jejich práci" or "za jejich příspěvky." or "za spolupráci, kteří se podíleli na tomto návrhu."

### AV(SAE-small(x))

Formal parliamentary document structure: official EU institutional language with structured speech format, presenting a resolution on the European Parliament's legislative process.

The sentence "Chci vyjádřit svou vděčnost členům Výboru za jejich pomoc a podporu" is a standard acknowledgment of the European Parliament, listing gratitude for the drafting process.

Final token "děkuje členům Parlamentu" ends a relative clause ("navíc bych se chtěla poděkovat..."), requiring a completion like "za jejich úsilí" or "za kritický názor." or "pro... za vynikající práci." or "za tuto otázku." specifically naming the MEPs.

### AV(SAE-big(x))

Formal parliamentary/institutional report structure: official EU legislative document tone, with structured description of a meeting summary in Slovak language.

The sentence "Ďakujeme poslancům Európskeho parlamentu a členom výboru, ktorí pomohli predsedovi a členom výboru" establishes acknowledgment of cooperation and gratitude.

Final token "predsedníkov" ends a noun phrase ("poďakovujeme poslancům a členom výboru..."), requiring a list of contributors or acknowledgments — likely "za ich pomoc" or "za jejich práci." or "za spolupráciu." or "v particular, pani X a Y."

## idx 125 — token `OP`

- doc/source/lang: `8699` / `FreeLaw` / `eng_Latn`
- selected by: `sae_big:fixed_point_leakage`#2
- KL: direct `1.17847e-07`, SAE-small `1.66867e-05`, small-loop `0.0304063`, SAE(direct)-small `7.8655e-06`, SAE-big `2.75344e-05`, big-loop `2.36894e-07`, SAE(direct)-big `4.76619e-06`

### Frozen context

```text
, Maymi, Rivera & Rotger, P.S.C., San Juan, PR, for Defendants.

OP
```

### Direct AV(x)

Legal document structure: bankruptcy court opinion format signals a standard ruling summary, with a federal judge's opinion on a motion to dismiss.

The phrase "ORDER" introduces a formal judicial opinion, requiring a ruling citation or case summary for the three named defendants (the Bank of America, Inc.).

Final token "OP" opens a court document heading ("OPINION AND ORDER"), immediately requiring a standard legal boilerplate like "INION" or "ORDER" to complete the heading, then likely "R BY THE COURT" or "DECISION ON MOTION TO DISMISS" or "THIS MEMORANDUM" or "RECOMMENDATION FOR THE CASE."

### AV(SAE-small(x))

Legal document format: structured case summary with formal boilerplate language, requiring a court opinion or motion to dismiss in a federal bankruptcy case.

The phrase "THE COURT ENTERED THE FOLLOWING OP" signals a standard federal court order citation, establishing the ruling on the motion for summary judgment against the parties.

Final token "Op'" opens a court order citation phrase ("Upon the Motion to Dismiss...Op. 1"), immediately requiring a legal document label like "ORDER" or "THE ISSUE OF THE CONTRACT" or "BY THE TRIAL COURT" or "1. IN THE CASE OF THE DEFENDANT" — a specific ruling or motion description referencing the parties' motion for judgment.

### AV(SAE-big(x))

Legal document structure: formal federal district court opinion format, establishing case summary for a civil rights action involving the Eastern District of Michigan.

The phrase "ORDER OF OPERATIONS: IN UNITED STATES DISTRICT COURT: OP" signals a standard judicial citation header, indicating a case ruling or motion document describing the appellate brief.

Final token "OP" opens a legal document heading ("ORDER OF OPERATIONS: OP..."), immediately requiring a noun phrase — most likely "Opinion" or "THE MOTION TO DISMISS" or "Order" — specifically the case caption or document type label, likely "by the District Court" or "THIS MEMORANDUM" or "Plaintiff's Motion."

## idx 127 — token ` value`

- doc/source/lang: `949` / `Github` / `code`
- selected by: `sae_small:high_fidelity_high_code_churn`#1
- KL: direct `0.0718198`, SAE-small `0.141189`, small-loop `0.142223`, SAE(direct)-small `0.101201`, SAE-big `0.0510158`, big-loop `0.201947`, SAE(direct)-big `0.0953964`

### Frozen context

```text
 to standard dot plots and histograms that incorporates aspects of both. The x-coordinate of a point is based on its exact value
```

### Direct AV(x)

Statistical visualization guidance is being established, explaining a chart type for a boxplot or strip chart showing quantitative data.

The sentence "The bars are aligned to the actual value of the data" sets up a conceptual mapping — the horizontal axis represents the raw value, implying a visual encoding principle.

Final token "value" ends a clause ("aligned to its actual value"), requiring a continuation like "in the dataset, while the histogram..." or "and the category is represented" or "minus the mean, making it a sorted histogram." or "whereas the input value is..." or "from the data, and the closest value is..." — a specific analogy to numeric rounding.

### AV(SAE-small(x))

Structured documentation pattern: technical statistical explanation establishing a conceptual framework, with a formal definition of boxplot visualization.

The sentence "The visualization focuses on the range of the target variable value" sets up a constraint about the boxplot, specifically the relationship between categorical data and the input values.

Final token "value" ends a relative clause ("represents the quantized value of the target variable"), requiring a predicate — likely "and the dataset mean," or "in the dataset, while the categorical variable represents..." or "within the histogram, making it a continuous representation" or "and, violating the assumption" or "of the data, while the aggregated..."

### AV(SAE-big(x))

Structured tutorial format establishes explanatory context: a statistical visualization walkthrough, guiding a conceptual understanding of boxplot construction.

The sentence "The y-axis uses the actual value of the dataset's own value" sets up a contrast — the visualization is mapping the input data against the boxplot, implying a key assumption about the categorical variable.

Final token "value" ends a clause ("is based on your dataset's actual value"), requiring a predicate — likely "and the bar chart data" or "in the dataset, while the other values..." or "from the dataset, making it comparable" or "and the corresponding values, while the X..." referencing the numeric input.

## idx 130 — token `ark`

- doc/source/lang: `3627` / `HackerNews` / `eng_Latn`
- selected by: `sae_big:fixed_point_leakage`#3
- KL: direct `0.96639`, SAE-small `0.584102`, small-loop `0.347301`, SAE(direct)-small `0.328443`, SAE-big `0.958251`, big-loop `0.0589444`, SAE(direct)-big `0.431154`

### Frozen context

```text
 unanswered questions regarding data usage and
price roadmap. What is their strategy after the first year?

~~~
matthewark
```

### Direct AV(x)

Technical documentation pattern: structured Q&A format with code snippets and descriptions, typical for a Linux/security tutorial.

The phrase "This function is called Matthewmark" signals a standard documentation boilerplate, suggesting a list of metadata tags or a description of the article.

Final token "mark" is part of a GitHub/documentation metadata tag ("matthewmark, the author is matthewark"), immediately requiring a username or name identifier — likely "e" or "1" to complete "matthewark", then likely "by" or "
" or "3" or "the blog post" to name the author or the `Microsoft.NET` category.

### AV(SAE-small(x))

Structured article format with numbered/listed content suggests a citation or reference entry is being provided, following the "AI/ML/NLP" pattern.

The phrase "Markus Arkk" signals a username or contributor handle, likely a username or profile name associated with the "Markus Arkham" or similar.

Final token "Markark" ends a username citation tag ("by modified by Matt Ark...marked by Matt Ark..."), immediately expecting a username handle or profile identifier like "e" or "ark" to complete the name, then likely "in the video" or "
" or "for the AI/creative/teaching context" to credit the author.

### AV(SAE-big(x))

Structured documentation pattern: technical AI/ML reference format with structured descriptions of a dataset, listing model metadata and capabilities.

The phrase "model name 'mark-to-ark'" signals a standard Wikipedia/Wikipedia-style term for "Mark's Mark" — a compound noun referencing the "Mark" prefix.

Final token "mark/ark'" opens a compound noun phrase ("model name: mark-to-ark..."), immediately requiring a noun like "room" or "by" to complete the idiom "model's mark" or "the creative/behavioral context" — likely "e" or "baker" or "with the user"

## idx 135 — token ` he`

- doc/source/lang: `5024` / `OpenWebText2` / `eng_Latn`
- selected by: `sae_big:high_fidelity_high_code_churn`#1
- KL: direct `0.278336`, SAE-small `0.49184`, small-loop `0.375643`, SAE(direct)-small `0.560716`, SAE-big `0.256311`, big-loop `0.686021`, SAE(direct)-big `0.684482`

### Frozen context

```text
’ve considered to be destructive, sometimes constructive. I wonder, you know, if he really is the man I thought he
```

### Direct AV(x)

Ongoing interview transcript format: a quote from Joe Biden is being reproduced, with a structured political commentary narrative about the military/immigration issue.

The phrase "he remembered the general he thought he was hiring, the man he thought he" sets up a direct quote about John Kelly's character or the idealized version of the general.

Final token "he" ends a relative clause ("the man he thought he"), requiring a verb phrase — most likely "was" or "would be," completing the quoted recollection about Kelly's military background or the initial perception. "would be" or "was" or "was, a hero" — the promised endorsement.

### AV(SAE-small(x))

Narrative structure: a character interview format established, with a story summary building toward a specific moment of doubt or revelation.

The phrase "he was questioning whether he remembered the kind of man he" signals a classic character arc — the protagonist's reluctance to embrace the cowboy identity or the missing courage.

Final token "he" ends a relative clause ("more aware of how heroic he...which he thought he"), requiring a verb phrase — likely "was" or "would be" or "was." or "was, back then." or "was supposed to be." or "was." — a specific past decision about the Captain's role or the promotion.

### AV(SAE-big(x))

Narrative structure: a character arc post is building toward a personal confession about a troubled protagonist, establishing emotional stakes.

The sentence "He was unsure if he wanted to become the man he admired, since he wondered if he felt he was as admirable as he" signals a psychological dilemma about the General's self-perception and the hiring decision.

Final token "he" ends a relative clause ("as admirable a person as he...believing he was the kind of person he"), requiring a verb phrase — likely "was" or "would be" or "was." or "was, especially given the Navy background" or "could be." to complete the thought.

## idx 153 — token `لاً`

- doc/source/lang: `9982` / `xnli:ar` / `ar`
- selected by: `sae_big:tiny_geometry_large_text_change`#1
- KL: direct `0.0699401`, SAE-small `0.0284743`, small-loop `0.102525`, SAE(direct)-small `0.115105`, SAE-big `0.0444556`, big-loop `0.103296`, SAE(direct)-big `0.0728325`

### Frozen context

```text
 مؤهلاً لاستيعاب الهزات المسبقة الزلزال. يبدو أن قلبك يكون مؤهلاً
```

### Direct AV(x)

Ongoing literary-philosophical essay structure: the text is presenting a poem with a repeating refrain ("الروح قادرة على تحمل الألم") and a concluding summary.

The sentence "يخبرك هذا القلب هو مؤهل" mirrors the earlier claim that the soul is "مناسبًا لاستيعاب الألم" — a thematic culmination.

Final token "مؤهلاً" ends a parallel clause ("هو مؤهلًا... هو مؤهلًا"), requiring a direct repetition or completion like "لاستيعاب الألم" or "لاستيعاب كل التحولات...". or "لاستيعاب الألم الذي يمر به." — the phrase echoing the soul's resilience.

### AV(SAE-small(x))

Structured article pattern: repeating structure of translated Arabic phrases with consistent formatting demands a concluding summary statement.

The sentence "السؤال الثاني هو ما هو الفرق بين النجاح والفشل ولقد وجدت إعدادًا مختلفًا" mirrors the parallel "أنا أجهز نفسي" theme.

Final token "التي..." ends a repeated parallel clause ("أجد خيارًا مناسبًا...والذي يجهز..."), requiring a direct parallel completion — "لت استقبال الصدمة" or "ل استقبال الصدمة" or "لت استقبال الصدمة إلى استقبال الصدمة" echoing the "لت استقبال الصدمة" for the soul/spirit category.

### AV(SAE-big(x))

Structured article pattern: repeating structure of translated/reproduced text with consistent formatting continues throughout the document.

The sentence "لقد أجد نفسي قادراً على تعلم الحب، ولكني لا أجد نفسي قادراً" mirrors the final question about the overarching theme of matching the protagonist's capacity to accommodate suffering.

Final token "قادر على" ends a parallel clause ("يظل العالم قادراً... يظل العالم قادراً") — requires a direct repetition or reformulation like "لت استقبال التدريب السفر" or "لاستقبال التدريب السفر إلى استقبال الألم" or "إلى استقبال التدريب... لتكرار نفس الهدف"

## idx 163 — token ` कोशिश`

- doc/source/lang: `10048` / `xnli:hi` / `hi`
- selected by: `sae_small:high_fidelity_high_code_churn`#3
- KL: direct `0.0441333`, SAE-small `0.117532`, small-loop `0.138331`, SAE(direct)-small `0.0382845`, SAE-big `0.0709379`, big-loop `0.0614629`, SAE(direct)-big `0.0506817`

### Frozen context

```text
 , अगले ही दिन उसे यूनाइटेड नेशंस ले गए । मैंने उसे पहले ही कह दिया था, मैंने उसे समजने की कोशिश
```

### Direct AV(x)

Article structure: Hindi astrology/Hindi forum format with answer-style content, presenting a Hindi quote about Indian paranormal/religious topic.

The sentence "मैंने आपको समझाने की कोशिश की थी कि मैं कोशिश करने की कोशिश" establishes a narrative of regret or failure, implying the speaker is recounting their efforts to warn or explain.

Final token "कोशिश" ends a clause ("मैंने पहले इंसान से बात करने की कोशिश"), requiring a verb phrase — likely "की थी" or "भी किया था" to complete the attempt statement, or "made" or "लेकिन वो भी किया" or "के साथ किया था" — a grammatical continuation about the past effort to explain.

### AV(SAE-small(x))

Structured answer format with Hindi/English question-answer pattern, requiring a Hindi poem or answer about a story/topic.

The sentence "आपने पहले घर की देखभाल करने की कोशिश की तो आपने पहले प्रयास में बहुत बड़ी गलती" sets up a list of actions the person did, implying a specific example of their interaction.

Final token "प्रयास" ends a clause ("आपने पहले कोशिश...तो आपने एक अलग कोशिश"), requiring a noun phrase — likely "की गई" or "कला थी" to complete the verb phrase, then "했다" or "के नाम से..." or "did... थी" to specify the action of doing the work/change in the social media post.

### AV(SAE-big(x))

Hindi/Indian language answer format: conversational tone with Hindi words like "mujhe maaf करना" and structured reply to a question.

The sentence "मैंने लोगो की परेशानी को ठीक करने की कोशिस की लेकिन मैं लोगो पर कोशिश करने की कोशिश" sets up a narrative of a failed/wrong action or a moral dilemma.

Final token "कोशिश" ends a verb phrase ("मैंने उसमे अलग काम की कोशिश..."), requiring a noun object — likely "की" or "किया" to complete the verb, then likely "कर ली थी" or "से कर ली" or "made" describing the attempt to do/did, referencing the poet's efforts.

## idx 168 — token ` Y`

- doc/source/lang: `10182` / `xnli:tr` / `tr`
- selected by: `sae_big:sae_grounding_code_rescue`#2, `sae_big:worst_sae_manifold_roundtrip`#1, `sae_small:sae_grounding_code_rescue`#1, `sae_small:worst_sae_manifold_roundtrip`#1
- KL: direct `0.939907`, SAE-small `1.01292`, small-loop `1.61648`, SAE(direct)-small `1.00512`, SAE-big `0.438629`, big-loop `0.629604`, SAE(direct)-big `1.49839`

### Frozen context

```text
 Ve, eğer yükselip dursaydı hızla geçerdi ve kafanı koparacakmış gibi olurdu. Y
```

### Direct AV(x)

Literary/creative writing site format: text is a collage of fragmented, humorous excerpts from a fantasy/fiction book, with absurdist prose.

The phrase "Durum's eyes were following you" continues a pattern of escalating, darkly humorous narrative commentary about the character's predicament.

Final token "Y" opens a new sentence fragment ("Y..."), likely continuing a quoted phrase or exclamation — "Yavaşla" or "the hill" or "they were going too fast" or "a new name for the repetition" or "But." or "so, they must stop." — a grammatical or poetic clause about the narrator's movement or the problem.

### AV(SAE-small(x))

Structured educational/personal blog format with numbered prompts and reflective writing style, continuing a list of mindfulness/creative writing topics.

The phrase "Yolunda'nın hedefi, duygusal yolculuğunuz hakkında düşünmek ve Y" signals a new question or topic sentence about the poem/advice.

Final token "Y" opens a repeated question prompt ("Yediğiniz yemekler...Y"), immediately requiring a topic phrase like "gerçek aşk" or "bir hikaye yaratmak" — likely "uzmanlık alanın" or "kendine..." or "uygun davranışlar olabilir" or "yolunuzda..." continuing the existential/music/nature theme.

### AV(SAE-big(x))

Fragmented, incoherent text pattern: scattered literary/word puzzle snippets suggest a word-association or writing prompt format with fragmented sentences.

The phrase "Yüz çevirmek" introduces a new thematic statement about the poem, implying a continuation of the narrative about the children's relationship with the box.

Final token "Y" opens a new clause ("Y..."), likely continuing with "the past" or a concept like "they are trapping the words" or "suddenly changed" — likely "bir boşluk" or "a new davranış." or "sometimes they ignore them." or "by dividing the landscape" or "like the thief's actions."

## idx 185 — token ` sell`

- doc/source/lang: `3557` / `Github` / `code`
- selected by: `sae_big:language_loop_rescue`#1, `sae_small:language_loop_catastrophe`#1
- KL: direct `16.3745`, SAE-small `1.7538e-05`, small-loop `19.9996`, SAE(direct)-small `23.4994`, SAE-big `0.00014504`, big-loop `4.01886`, SAE(direct)-big `11.1256`

### Frozen context

```text
 without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
```

### Direct AV(x)

Software license agreement boilerplate pattern: standard Apache License text is being reproduced, following a structured legal document format for the JAXR API.

The phrase "or to sublicense, sell, lease, or sell" is a standard copyright clause listing permitted uses of the Software, mirroring the Apache License 2.0 boilerplate.

Final token "sell" ends a list of verb phrases ("or to sell...or to distribute copies of the Software, or to sell"), requiring immediate continuation with "copies of the Software" or "copies of the License" or "the License, subject to the Agreement" or "a derivative thereof" — the object of the verb phrase.

### AV(SAE-small(x))

Structured legal/medical disclaimer format: boilerplate Apache License boilerplate is being reproduced, following standard software documentation conventions for the Nutrition Data API.

The phrase "This software is not a substitute for professional medical advice. You may not sell..." signals a standard license boilerplate listing prohibited uses and commercial restrictions.

Final token "sell" is part of a legal copyright phrase ("including the right to sell, distribute, or modify..."), requiring a noun phrase like "copies of the Software" or "copies of the Health Declaration" — likely "copies of the Software or any derivative thereof" or "a physician or copies of the Health Declaration" to complete the licensing clause for the practitioner.

### AV(SAE-big(x))

Legal disclaimer boilerplate pattern: structured template for a website privacy notice, following standard copyright/licensing language for the HIPAA Compliance Toolkit.

The phrase "This website is not intended for sale or redistribution, and you are not permitted to sell or sell" establishes a standard license restriction clause for the Terms of Service.

Final token "sell" ends a repeated clause ("or to sell...the copy of this Agreement..."), requiring a parallel verb phrase — likely "a copy of the Software" or "
Copies of the Software" or "it to the United States." or "by the Software, Inc." or "a copy of this Work, for personal use only."

## idx 186 — token ` BE`

- doc/source/lang: `3557` / `Github` / `code`
- selected by: `sae_big:language_code_leakage`#2, `sae_big:language_loop_catastrophe`#3, `sae_small:language_loop_rescue`#1, `sae_small:worst_sae_manifold_roundtrip`#3
- KL: direct `6.50398`, SAE-small `4.35919e-09`, small-loop `4.6015e-05`, SAE(direct)-small `10.75`, SAE-big `3.52798e-06`, big-loop `13.2682`, SAE(direct)-big `6.2635`

### Frozen context

```text
 OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE
```

### Direct AV(x)

Legal disclaimer boilerplate format: structured document listing liability limitations for a software copyright notice, following standard Apache License phrasing.

The phrase "The authors or copyright holders shall not be liable for any damages or losses arising from the fonts or the copyright holders be" signals a standard legal liability clause typical in open-source documentation.

Final token "BE" ends a legal liability phrase ("The authors or copyright holders shall not be..."), requiring a verb phrase like "liable for any damages" or "held liable for the act of serving this document" or " LIABLE FOR THE TORT OF PROVIDING THIS SOFTWARE." or "in any way for the action of serving..." to complete the clause.

### AV(SAE-small(x))

Legal disclaimer boilerplate format: structured listing of standard liability disclaimers for a software/website, following a predictable template for copyright notice.

The phrase "not liable for any damages or losses, including indirect, incidental, or consequential damages" is a standard legal disclaimer clause, requiring a standard liability limitation phrase.

Final token "be" opens a liability clause ("not liable for any... damages...not be liable to any third parties"), requiring a legal phrase like "LIABLE FOR THIS SITE" or "ANY WARRANTY OR LIABILITY ARISING FROM THIS SOFTWARE" or "THE LIABILITY OF THE AUTHOR OR PROVIDER" or "SPECIFIC MEDICAL ADVICE OR DAMAGES CAUSED BY."

### AV(SAE-big(x))

Structured legal document format: boilerplate disclaimer boilerplate is being followed, with a standard HTML/contract template for a personal injury/liability page.

The phrase "developers will not be held liable" establishes a liability disclaimer clause, requiring a specific legal consequence statement about the negligence lawsuit against the defendant.

Final token "be" ends a liability clause ("will not be held liable..."), requiring a verb phrase — most likely "liable for any damages" or "sued by the User" or "responsible for the Code" or "liability...in the event of any incident." or "...in the lawsuit." — a standard legal boilerplate phrase for the Bootstrap/HTML.

## idx 189 — token ` adequate`

- doc/source/lang: `4716` / `NIH ExPorter` / `eng_Latn`
- selected by: `sae_big:high_fidelity_high_code_churn`#2
- KL: direct `0.030654`, SAE-small `0.0924919`, small-loop `0.329869`, SAE(direct)-small `0.0997104`, SAE-big `0.093161`, big-loop `0.206762`, SAE(direct)-big `0.106757`

### Frozen context

```text
 causative model. Unless we use an integrated approach, grounded in a unifying public health model, we will not have taken the adequate
```

### Direct AV(x)

Academic public health paper structure: formal policy/research article tone with abstract summarizing epidemiological burden of chronic disease in Canada.

The sentence "we have not taken the adequate" is building a concluding argument about the need for comprehensive interventions, urging the reader toward the gap between current knowledge and actionable prevention strategies.

Final token "adequate" ends a clause ("has not taken the adequate"), requiring a noun phrase — likely "steps/measures/actions to address the problem" or "steps to prevent the epidemic" or "necessary interventions to achieve this" or "effort to address the complexity." or "first step toward translating the findings into effective solutions" to complete the rhetorical statement.

### AV(SAE-small(x))

Academic policy/health discourse structure: formal argumentative framing with structured conclusion urging systemic resilience against TB epidemic in a developing nation context.

The sentence "Is this the right approach to take the appropriate" sets up a concluding statement about the need for comprehensive, evidence-based interventions to prevent the spread of health inequity.

Final token "take the appropriate" opens a noun phrase ("to take the...in order to take the appropriate"), requiring a noun like "steps/actions" or "policy measures" — likely "action to address the crisis" or "moral responsibility of the healthcare system" or "steps to safeguard the system" or "social value framework to achieve..."

### AV(SAE-big(x))

Academic/policy framework structure: formal EU health research paper tone, establishing structured argument around road safety intervention in occupational health context.

The sentence "does not take the necessary" sets up a concluding statement about the gap between the problem and the intervention, implying the health system lacks sufficient action or solutions.

Final token "necessary" ends a noun phrase ("not taking the appropriate...does not take the necessary"), requiring a noun phrase completion — likely "steps/actions" or "measures to address the issue" or "steps to achieve the outcome." or "first steps towards resolving it" or "actions by the stakeholders" — the full hierarchy of intervention or targeted support.

## idx 191 — token ` all`

- doc/source/lang: `1287` / `OpenSubtitles` / `eng_Latn`
- selected by: `sae_big:language_loop_rescue`#2, `sae_small:language_loop_rescue`#3
- KL: direct `1.76298`, SAE-small `0.453011`, small-loop `0.97536`, SAE(direct)-small `3.51727`, SAE-big `0.0428673`, big-loop `0.186044`, SAE(direct)-big `2.57783`

### Frozen context

```text
't." "WIND HOWLS" " So, what's for lunch?" " Stew." "Is that all
```

### Direct AV(x)

Dialogue format with a waitress, following a pattern of conversational exchanges with a rural diner customer.

The phrase "Is that all" signals a rhetorical question about the menu, building toward a sarcastic or grim comment about the limited lunch options — "meatloaf or all" implies a question about the meal.

Final token "all" ends a quoted question fragment ("Is that all...or is there only meatloaf? Is that all"), requiring a closing question mark or continuation like "you're having?" or "there's?" or "we eat?" or "there's a meat?" — likely a negative dietary option or a question about the meal's contents.

### AV(SAE-small(x))

Dialogue format with structured conversational exchanges establishes a humorous, playful tone — a character is narrating a fictional restaurant scene.

The sentence "So, is that all...are you hungry? Is everything all" sets up a question about the food, implying a rhetorical question or a playful remark about the menu/experience.

Final token "all" ends a quoted question fragment ("Is that all, or is it all..."), requiring a closing clause — likely "you have?" or "there?" or "you're eating?" or "it?" or "there's a special kind of dessert?" — completing the question about the bear's request or the mysterious food.

### AV(SAE-big(x))

Dialogue format with a character speaking, establishing a conversational pattern of frustration and a rhetorical question about eating.

The sentence "So you're just asking...is there nothing on the menu...all small portions? Or is it all" sets up a sarcastic, dismissive closing statement about the meal.

Final token "all" ends a rhetorical question fragment ("Is there nothing on the menu...or is it all..."), requiring a closing clause like "you've been eating?" or "there?" or "you're serving?" or "of the boring food?" — a question about the character's dietary choices or a question about the single meat/food.

## idx 192 — token ` what`

- doc/source/lang: `1287` / `OpenSubtitles` / `eng_Latn`
- selected by: `sae_big:high_fidelity_high_code_churn`#3
- KL: direct `0.0871145`, SAE-small `0.205401`, small-loop `0.384018`, SAE(direct)-small `0.452507`, SAE-big `0.0815404`, big-loop `0.561806`, SAE(direct)-big `0.193773`

### Frozen context

```text
s for lunch?" " Stew." "Is that all you can cook?" "Give me some money and I can see what
```

### Direct AV(x)

Dialogue format with a character speaking in a casual, hungry tone — a cook narrating a food quest in a small town.

The sentence "I'll see what" sets up a playful, sarcastic closing line about the cooking/ordering options, implying a self-deprecating search for a special meal.

Final token "what" ends a subordinate clause ("see what...I'll see what"), requiring a verb phrase — likely "I can make" or "else I can find" or "the chef can do." or "I get." or "special I can order." — completing the thought about the culinary possibilities or a new skill/ingredient discovery.

### AV(SAE-small(x))

Conversational food blog register established, with casual tone and playful commentary about a cooking/dining post.

The sentence "So I need to look up what" signals a tentative, humorous request for a fallback meal suggestion, implying a brainstorming moment about the restaurant discovery or a new recipe idea.

Final token "what" opens a relative clause ("see what...or try to figure out what"), requiring a noun phrase — likely "I'm making" or "other food I can find" or "the kitchen offers" or "I'd try." or "kind of cooking I do." or "I'm ordering/serving" referencing the menu or mood.

### AV(SAE-big(x))

Conversational foodie/lifestyle blog tone established, with playful banter and casual UK voice ("Hey, Chef!") guiding a humorous post.

The sentence structure "I wanted to see what" sets up a self-deprecating, hungry confession about exploring the menu or ordering more gourmet food.

Final token "what" ends a subordinate clause ("see what...then try to figure out what"), requiring a verb phrase — likely "I can make" or "else I can get" or "the kitchen can do" or "I'd try." or "I can find nearby" or "else I need to order/explore." referencing the cooking/food topic.

