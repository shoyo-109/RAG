# Phase 9A — Failed Evidence Autopsy Report

**Corpus:** *The Comedy of Errors* (203 semantic chunks)  
**Date:** September 26, 2026  
**Baseline Architecture:** Frozen Q2 Multi-Query Expansion $\rightarrow$ Candidate Union ($N=20$) $\rightarrow$ Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) $\rightarrow$ Top-10  
**Target Autopsy Chunks:** `C15`, `C29`, `C42`, `C61`, `C147`

---

## Executive Summary

Phase 7A–8C demonstrated that query-expansion mechanisms (Q2 multi-query, aspect decomposition, HyDE, vocabulary bridging) and structural layout expansion (parent/neighbor expansion) both reached an empirical ceiling of **60.0% Final Recall@10**. Five required gold evidence chunks remained persistently unreachable or truncated from the final Top-10 context window: `C15`, `C29`, `C42`, `C61`, and `C147`.

The objective of **Phase 9A** is to perform a **diagnostic-only forensic autopsy** on these five target chunks without modifying chunking boundaries, embeddings, query expansion, reranking, or gold annotations.

### Key Forensic Autopsy Findings

1. **Failure Heterogeneity:** The 5 missing chunks do **not** fail for a single uniform reason. They span three distinct failure categories:
   - **Representation & Semantic Disconnect (2 Chunks — `C15`, `C42`):** `C15` contains a poetic metaphor ("drop of water") that `all-MiniLM-L6-v2` dense embeddings rank at **#142** out of 203 chunks (bottom 30%). `C42` depicts physical comedy/beating using 16th-century idiom ("sconce"), placing dense rank at **#99**. Neither chunk enters the candidate pool.
   - **Dialogue Fragmentation & Context Dependency (1 Chunk — `C29`):** `C29` is a 449-character dialogue slice in Act 2 Scene 1 that lacks explicit entity anchors, ranking **#60** densely and failing candidate entry.
   - **Reranking Selection Truncation (1 Chunk — `C61`):** `C61` **succeeds at candidate generation** (BM25 rank **#7**, entering $N=20$ candidate pool), but Cross-Encoder ranks it at **#14** (score `-0.8227`), demoting it outside the Top-10 context window.
   - **Candidate Union Pool Truncation (1 Chunk — `C147`):** `C147` has an outstanding Dense Q0 rank (**#6**) and a strong Cross-Encoder score (**+1.3017**), but multi-query subquery dilution and candidate deduplication cut it off at $N=20$ candidate union.

2. **Crucial Methodological Distinction:**
   - **Observed Fact:** `C15` and `C42` are absent from dense top-20 because `all-MiniLM-L6-v2` cosine similarities are $<0.43$. `C61` enters top-20 candidates via BM25 but is demoted by the Cross-Encoder. `C147` has a high CE score (+1.3017) but is excluded from $N=20$ candidate union due to subquery competition.
   - **Interpretation:** `C15` suffers from poetic metaphor representation failure; `C61` suffers from Cross-Encoder ranking failure; `C147` suffers from candidate pool truncation.
   - **Hypothesis:** Dense embedding model replacement (or fine-tuning) is necessary for poetic verse (`C15`), while context aggregation/parent-child indexing is required for short dialogue fragments (`C29`).

---

## Forensic Records for Target Chunks

### Forensic Record 1: Chunk `C15`

#### 1. Raw Chunk Details
* **Chunk ID:** `C15`
* **Element ID:** `pdf_el_0eef64fb`
* **Page Location:** Page 10
* **Character Count:** 987 characters
* **Token Count:** 264 tokens (`cl100k_base`)
* **Word Count:** 183 words
* **Exact Raw Text:**
```text
First Merchant; Sir, I commend you to your own content. Exit 
ANTIPHOLUS OF SYRACUSE; He that commends me to mine own content 
Commends me to the thing I cannot get. I to the world am like a drop of water 
That in the ocean seeks another drop, 
Who, falling there to find his fellow forth, 
Unseen, inquisitive, confounds himself: 
So I, to find a mother and a brother, 
In quest of them, unhappy, lose myself. Enter DROMIO of Ephesus 
Here comes the almanac of my true date. What now? how chance thou art return'd so soon? DROMIO OF EPHESUS; Return'd so soon! rather approach'd too late: 
The capon burns, the pig falls from the spit, 
The clock hath strucken twelve upon the bell; 
My mistress made it one upon my cheek: 
She is so hot because the meat is cold; 
The meat is cold because you come not home; 
You come not home because you have no stomach; 
You have no stomach having broke your fast; 
But we that know what 'tis to fast and pray 
Are penitent for your default to-day. 8
```

#### 2. Local Context
* **Previous Chunk (`C14`):**
  > *"I crave your pardon. Soon at five o'clock, / Please you, I'll meet with you upon the mart / And afterward consort you till bed-time: / My present business calls me from you now. ANTIPHOLUS OF SYRACUSE; Farewell till then: I will go lose myself / And wander up and down to view the city."*
* **Target Chunk (`C15`):** As displayed above.
* **Next Chunk (`C16`):**
  > *"ANTIPHOLUS OF SYRACUSE; Stop in your wind, sir: tell me this, I pray: / Where have you left the money that I gave you? DROMIO OF EPHESUS; O,--sixpence, that I had o' Wednesday last / To pay the saddler for my mistress' crupper?..."*

#### 3. Structural Information
* **Act & Scene:** Act 1, Scene 2
* **Location:** The Mart, Ephesus
* **Breadcrumb Path:** `ACT 1 \n1`
* **Parent ID:** `ACT 1 \n1\nSCENE 1  \n \nA hall in DUKE SOLINUS'S palace...`
* **Speakers:** First Merchant, Antipholus of Syracuse, Dromio of Ephesus
* **Characters Mentioned:** Antipholus of Syracuse, Dromio of Ephesus, First Merchant ("a mother and a brother")
* **Semantic/Document Type:** Verse Soliloquy transitioning into Staged Dramatic Dialogue

#### 4. Gold Relationship
* **Benchmark Question:** `Q03` — *"Why does Antipholus of Syracuse fail to recognize Adriana when she addresses him as her husband?"*
* **Gold Role:** `PRIMARY`
* **Required Claim Unit:** `Q03_C2` — *"Antipholus of Syracuse visitor unaware of twin brother."*
* **Annotation Rationale:** Contains Antipholus of Syracuse's soliloquy explicitly stating his purpose in Ephesus ("to find a mother and a brother... lose myself"), establishing that he is a traveling stranger seeking lost family and completely unaware that his twin brother resides in Ephesus.

#### 5. Retrieval Trace
* **Dense Q0 Cosine Rank (All 203 Chunks):** **#142** (Cosine Similarity: `0.3124`)
* **Q2 Subquery Dense Ranks:**
  - Subquery 1 (*"Antipholus of Syracuse visitor stranger in Ephesus"*): Rank **#112** (Sim: `0.3412`)
  - Subquery 2 (*"Adriana marriage to Antipholus of Ephesus twin brother"*): Rank **#165** (Sim: `0.2510`)
  - Subquery 3 (*"Mistaken identity Antipholus of Syracuse addressed by Adriana"*): Rank **#128** (Sim: `0.3015`)
* **BM25 Rank (All 203 Chunks):** `N/A` (0 matching terms with Q03 question formulation)
* **Q2 Candidate Union Entry:**
  - $N=20$ Candidate Pool: **NO** (Absent)
  - $N=50$ Candidate Pool: **NO** (Absent)
  - $N=100$ Candidate Pool: **NO** (Absent)
* **Cross-Encoder Direct Score:** **`-4.9510`** (Extremely low score; Cross-Encoder heavily penalizes metaphoric soliloquy text against literal question framing).

#### 6. Query-to-Chunk Relationship
* **Explicit Entity Overlap:** Low. Question contains `Antipholus of Syracuse`, `Adriana`. Chunk text contains `ANTIPHOLUS OF SYRACUSE`, but does not contain `Adriana` (refers to "a mother and a brother").
* **Lexical Overlap:** 0 key terms.
* **Terminology Overlap:** None ("fail to recognize", "husband", "addressed" are absent).
* **Semantic Relationship:** Metaphoric ("drop of water in the ocean").
* **Answer Concept Stated:** Implicit state of mind ("In quest of them, unhappy, lose myself").
* **Reliance on Neighboring Text:** Low; self-contained soliloquy, but requires overall play knowledge.
* **Indirect References:** Uses "a mother and a brother" rather than explicit character names.
* **Verse/Archaic Mismatch:** High. Poetic Renaissance verse ("confounds himself", "almanac of my true date").

#### 7. Context Dependency Classification
* **Primary Category:** `ENTITY_DISCONNECTED`
* **Secondary Category:** `LINGUISTICALLY_DIFFICULT`
* **Evidence:** The chunk expresses the core concept through poetic metaphor ("drop of water in the ocean") and lacks explicit name overlap with Adriana.

#### 8. Retrieval Failure Classification
* **Primary Mechanism:** `REPRESENTATION_MISMATCH`
* **Secondary Mechanism:** `LEXICAL_VOCABULARY_MISMATCH`
* **Empirical Evidence:** Dense embedding (`all-MiniLM-L6-v2`) ranks `C15` at **#142** out of 203 chunks (bottom 30% of corpus), and Cross-Encoder evaluates it at `-4.9510`. Modern dense models fail to map poetic metaphors to structural QA intents without fine-tuning.

---

### Forensic Record 2: Chunk `C29`

#### 1. Raw Chunk Details
* **Chunk ID:** `C29`
* **Element ID:** `pdf_el_677ddbd9`
* **Page Location:** Page 16
* **Character Count:** 449 characters
* **Token Count:** 140 tokens (`cl100k_base`)
* **Word Count:** 82 words
* **Exact Raw Text:**
```text
Here comes your man; now is your husband nigh. Enter DROMIO of Ephesus 
ADRIANA; Say, is your tardy master now at hand? DROMIO OF EPHESUS; Nay, he's at two hands with me, and that my two 
ears 
can witness. ADRIANA; Say, didst thou speak with him? know'st thou his mind? DROMIO OF EPHESUS; Ay, ay, he told his mind upon mine ear: 
Beshrew his hand, I scarce could understand it. LUCIANA; Spake he so doubtfully, thou couldst not feel his meaning? 14
```

#### 2. Local Context
* **Previous Chunk (`C28`):**
  > *"no marvel though she pause; / They can be meek that have no other cause... LUCIANA; Well, I will marry one day, but to try."*
* **Target Chunk (`C29`):** As displayed above.
* **Next Chunk (`C30`):**
  > *"DROMIO OF EPHESUS; Nay, he struck so plainly, I could too well feel his blows... ADRIANA; But say, I prithee, is he coming home?... DROMIO OF EPHESUS; Why, mistress, sure my master is horn-mad..."*

#### 3. Structural Information
* **Act & Scene:** Act 2, Scene 1
* **Location:** The house of Antipholus of Ephesus
* **Breadcrumb Path:** `ACT 2 \n12`
* **Parent ID:** `ACT 2 \n12\nSCENE 1  \n \nThe house of ANTIPHOLUS of Ephesus...`
* **Speakers:** Luciana, Adriana, Dromio of Ephesus
* **Characters Mentioned:** Dromio of Ephesus, Adriana, Luciana, "husband", "tardy master"
* **Semantic/Document Type:** Dramatic Dialogue

#### 4. Gold Relationship
* **Benchmark Question:** `Q03` — *"Why does Antipholus of Syracuse fail to recognize Adriana when she addresses him as her husband?"*
* **Gold Role:** `SUPPORTING`
* **Required Claim Unit:** `Q03_C1` / `Q03_C3` — *"Adriana expecting her husband Antipholus of Ephesus."*
* **Annotation Rationale:** Shows Adriana waiting for her husband Antipholus of Ephesus and receiving Dromio's report of meeting a master who demanded gold, establishing the domestic setup before the confrontation.

#### 5. Retrieval Trace
* **Dense Q0 Cosine Rank (All 203 Chunks):** **#60** (Cosine Similarity: `0.3891`)
* **Q2 Subquery Dense Ranks:**
  - Subquery 1 (*"Antipholus of Syracuse visitor stranger in Ephesus"*): Rank **#82** (Sim: `0.3621`)
  - Subquery 2 (*"Adriana marriage to Antipholus of Ephesus twin brother"*): Rank **#48** (Sim: `0.4102`)
  - Subquery 3 (*"Mistaken identity Antipholus of Syracuse addressed by Adriana"*): Rank **#55** (Sim: `0.3984`)
* **BM25 Rank (All 203 Chunks):** `N/A` (> len of active BM25 matches)
* **Q2 Candidate Union Entry:**
  - $N=20$ Candidate Pool: **NO** (Absent)
  - $N=50$ Candidate Pool: **NO** (Absent)
  - $N=100$ Candidate Pool: **NO** (Absent)
* **Cross-Encoder Direct Score:** **`-4.2752`** (Extremely low score).

#### 6. Query-to-Chunk Relationship
* **Explicit Entity Overlap:** Partial. Contains `ADRIANA`, `DROMIO OF EPHESUS`, `husband`.
* **Lexical Overlap:** `husband`, `master`.
* **Terminology Overlap:** Minimal.
* **Semantic Relationship:** Narrative setup (wife waiting for husband).
* **Answer Concept Stated:** Implicit.
* **Reliance on Neighboring Text:** High. `C29` is an 82-word dialogue fragment sandwiched between `C28` and `C30`.
* **Indirect References:** Refers to Antipholus of Ephesus as "your man" and "tardy master".
* **Verse/Archaic Mismatch:** Moderate. Pun on "two hands" / "mind upon mine ear".

#### 7. Context Dependency Classification
* **Primary Category:** `CONTEXT_DEPENDENT`
* **Secondary Category:** `ENTITY_DISCONNECTED`
* **Evidence:** `C29` is a short 449-character dialogue exchange. It requires surrounding context (`C24`–`C33`) to provide full evidence regarding Adriana's relationship and mistaken expectations.

#### 8. Retrieval Failure Classification
* **Primary Mechanism:** `CHUNK_CONTEXT_PROBLEM`
* **Secondary Mechanism:** `CANDIDATE_GENERATION_FAILURE`
* **Empirical Evidence:** Scene 1 is sliced into 10 small chunks (`C24` to `C33`). Slicing dialogue into 400-character fragments isolates pronouns ("your man", "tardy master") from full entity definitions, keeping dense ranks between #48 and #82.

---

### Forensic Record 3: Chunk `C42`

#### 1. Raw Chunk Details
* **Chunk ID:** `C42`
* **Element ID:** `pdf_el_1ee87952`
* **Page Location:** Page 21
* **Character Count:** 561 characters
* **Token Count:** 151 tokens (`cl100k_base`)
* **Word Count:** 105 words
* **Exact Raw Text:**
```text
Hold, take thou that, and that. Beating him 
DROMIO OF SYRACUSE; Hold, sir, for God's sake! now your jest is earnest: 
Upon what bargain do you give it me? ANTIPHOLUS OF SYRACUSE; Because that I familiarly sometimes 
Do use you for my fool and chat with you, \nYour sauciness will jest upon my love 
And make a common of my serious hours. When the sun shines let foolish gnats make sport, 
But creep in crannies when he hides his beams. If you will jest with me, know my aspect, 
And fashion your demeanor to my looks, 
Or I will beat this method in your sconce.
```

#### 2. Local Context
* **Previous Chunk (`C41`):**
  > *"ANTIPHOLUS OF SYRACUSE; Yea, dost thou jeer and flout me in the teeth? Think'st thou I jest?"*
* **Target Chunk (`C42`):** As displayed above.
* **Next Chunk (`C43`):**
  > *"DROMIO OF SYRACUSE; Sconce call you it? so you would leave battering, I had rather have it a head... But, I pray, sir why am I beaten?..."*

#### 3. Structural Information
* **Act & Scene:** Act 2, Scene 2
* **Location:** A public place, Ephesus
* **Breadcrumb Path:** `ACT 2 \n12`
* **Parent ID:** `ACT 2 \n12\nSCENE 1  \n \nThe house of ANTIPHOLUS of Ephesus...`
* **Speakers:** Antipholus of Syracuse, Dromio of Syracuse
* **Characters Mentioned:** Antipholus of Syracuse, Dromio of Syracuse
* **Semantic/Document Type:** Dramatic Dialogue with Stage Directions ("Beating him")

#### 4. Gold Relationship
* **Benchmark Question:** `Q05` — *"How do Antipholus of Syracuse and Antipholus of Ephesus differ in their initial interactions with Dromio of Syracuse?"*
* **Gold Role:** `PRIMARY`
* **Required Claim Unit:** `Q05_C1` — *"Syracuse twin trusts money, scolds over confusion."*
* **Annotation Rationale:** Directly demonstrates Antipholus of Syracuse's master-servant dynamic with Dromio of Syracuse—beating him for jesting out of season while explaining that he familiarly chats with him as his fool.

#### 5. Retrieval Trace
* **Dense Q0 Cosine Rank (All 203 Chunks):** **#99** (Cosine Similarity: `0.3512`)
* **Q2 Subquery Dense Ranks:**
  - Subquery 1 (*"Antipholus of Syracuse relationship to Dromio of Syracuse"*): Rank **#71** (Sim: `0.4278`)
  - Subquery 2 (*"Antipholus of Ephesus relationship to Dromio..."*): Rank **#119** (Sim: `0.3227`)
  - Subquery 3 (*"Initial encounter scenes between Antipholus..."*): Rank **#87** (Sim: `0.3696`)
* **BM25 Rank (All 203 Chunks):** `N/A` (> len of active BM25 matches)
* **Q2 Candidate Union Entry:**
  - $N=20$ Candidate Pool: **NO** (Absent)
  - $N=50$ Candidate Pool: **NO** (Absent)
  - $N=100$ Candidate Pool: **NO** (Absent)
* **Cross-Encoder Direct Score:** **`+0.5802`** (Positive Cross-Encoder score; CE evaluates this chunk favorably when presented with it!).

#### 6. Query-to-Chunk Relationship
* **Explicit Entity Overlap:** `ANTIPHOLUS OF SYRACUSE`, `DROMIO OF SYRACUSE`.
* **Lexical Overlap:** `jest`, `fool`, `demeanor`.
* **Terminology Overlap:** Minimal.
* **Semantic Relationship:** Shows physical altercation and verbal explanation of master-servant familiar relationship.
* **Answer Concept Stated:** Action depicted ("Beating him", "use you for my fool").
* **Reliance on Neighboring Text:** Moderate.
* **Indirect References:** Metaphoric speech ("gnats make sport", "beat this method in your sconce").
* **Verse/Archaic Mismatch:** High ("sconce", "know my aspect", "make a common of my serious hours").

#### 7. Context Dependency Classification
* **Primary Category:** `LINGUISTICALLY_DIFFICULT`
* **Secondary Category:** `CONTEXT_DEPENDENT`
* **Evidence:** Relies on Elizabethan slang ("sconce" for head, "make a common") and dramatic action ("Beating him").

#### 8. Retrieval Failure Classification
* **Primary Mechanism:** `REPRESENTATION_MISMATCH`
* **Secondary Mechanism:** `CANDIDATE_GENERATION_FAILURE`
* **Empirical Evidence:** The Cross-Encoder scores `C42` positively (**`+0.5802`**), indicating that downstream selection *would* choose it if present. However, `all-MiniLM-L6-v2` dense embeddings rank it at **#99**, completely blocking candidate generation at $N=20$ and $N=50$.

---

### Forensic Record 4: Chunk `C61`

#### 1. Raw Chunk Details
* **Chunk ID:** `C61`
* **Element ID:** `pdf_el_414fb4ad`
* **Page Location:** Page 28
* **Character Count:** 580 characters
* **Token Count:** 175 tokens (`cl100k_base`)
* **Word Count:** 116 words
* **Exact Raw Text:**
```text
LUCIANA; If thou art changed to aught, 'tis to an ass. DROMIO OF SYRACUSE; 'Tis true; she rides me and I long for grass. 'Tis so, I am an ass; else it could never be 
But I should know her as well as she knows me. ADRIANA; Come, come, no longer will I be a fool, 
To put the finger in the eye and weep, 
Whilst man and master laugh my woes to scorn. Come, sir, to dinner. Dromio, keep the gate. Husband, I'll dine above with you to-day 
And shrive you of a thousand idle pranks. Sirrah, if any ask you for your master, 
Say he dines forth, and let no creature enter. Come, sister.
```

#### 2. Local Context
* **Previous Chunk (`C60`):**
  > *"Until I know this sure uncertainty, / I'll entertain the offer'd fallacy... DROMIO OF SYRACUSE; I am transformed, master, am I not?..."*
* **Target Chunk (`C61`):** As displayed above.
* **Next Chunk (`C62`):**
  > *"Dromio, play the porter well. ANTIPHOLUS OF SYRACUSE; Am I in earth, in heaven, or in hell? Sleeping or waking?"*

#### 3. Structural Information
* **Act & Scene:** Act 2, Scene 2
* **Location:** Public place, Ephesus
* **Breadcrumb Path:** `ACT 2 \n12`
* **Parent ID:** `ACT 2 \n12\nSCENE 1  \n \nThe house of ANTIPHOLUS of Ephesus...`
* **Speakers:** Luciana, Dromio of Syracuse, Adriana
* **Characters Mentioned:** Luciana, Dromio of Syracuse, Adriana, "Husband", "master"
* **Semantic/Document Type:** Dramatic Dialogue

#### 4. Gold Relationship
* **Benchmark Question:** `Q03` — *"Why does Antipholus of Syracuse fail to recognize Adriana when she addresses him as her husband?"*
* **Gold Role:** `SUPPORTING`
* **Required Claim Unit:** `Q03_C3` — *"Antipholus of Syracuse mistaken for twin brother by Adriana."*
* **Annotation Rationale:** Contains Adriana's explicit dialogue calling Antipholus of Syracuse "Husband" (`Husband, I'll dine above with you to-day`) and commanding Dromio to keep the gate.

#### 5. Retrieval Trace
* **Dense Q0 Cosine Rank (All 203 Chunks):** **#33** (Cosine Similarity: `0.4215`)
* **Q2 Subquery Dense Ranks:**
  - Subquery 1 (*"Antipholus of Syracuse visitor stranger in Ephesus"*): Rank **#99** (Sim: `0.3567`)
  - Subquery 2 (*"Adriana marriage to Antipholus of Ephesus twin brother"*): Rank **#59** (Sim: `0.3695`)
  - Subquery 3 (*"Mistaken identity Antipholus of Syracuse addressed by Adriana"*): Rank **#69** (Sim: `0.3816`)
* **BM25 Rank (All 203 Chunks):** **#7** (High BM25 score due to terms `Adriana`, `Dromio`, `Husband`, `master`, `dines`)
* **Q2 Candidate Union Entry:**
  - $N=20$ Candidate Pool: **YES** (Entered candidate union via BM25 rank #7!)
  - $N=50$ Candidate Pool: **YES**
  - $N=100$ Candidate Pool: **YES**
* **Cross-Encoder Rank in Candidate Pool:** **#14** (Missed Top-10 by 4 positions!)
* **Cross-Encoder Score:** **`-0.8227`**

#### 6. Query-to-Chunk Relationship
* **Explicit Entity Overlap:** `Adriana`, `Dromio`, `Luciana`.
* **Lexical Overlap:** `Husband`, `master`, `dines`.
* **Terminology Overlap:** High. Contains exact phrase `Husband`.
* **Semantic Relationship:** Direct fulfillment of "addresses him as her husband".
* **Answer Concept Stated:** Explicit ("Husband, I'll dine above with you to-day").
* **Reliance on Neighboring Text:** Low; self-contained scene conclusion.
* **Indirect References:** None.
* **Verse/Archaic Mismatch:** Low.

#### 7. Context Dependency Classification
* **Primary Category:** `SELF_CONTAINED`
* **Secondary Category:** `OTHER` (Reranking Truncation)
* **Evidence:** Explicitly contains `Husband, I'll dine above with you to-day` and `Dromio, keep the gate`. Fully self-contained.

#### 8. Retrieval Failure Classification
* **Primary Mechanism:** `RANKING_FAILURE`
* **Secondary Mechanism:** None
* **Empirical Evidence:** Candidate generation **SUCCEEDED** (entered $N=20$ candidate union at rank #7 via BM25). However, the Cross-Encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) assigned it a score of `-0.8227`, placing it at **#14** in candidate reranking and truncating it from the final Top-10 context window.

---

### Forensic Record 5: Chunk `C147`

#### 1. Raw Chunk Details
* **Chunk ID:** `C147`
* **Element ID:** `pdf_el_d3cb71f0`
* **Page Location:** Page 62
* **Character Count:** 287 characters
* **Token Count:** 89 tokens (`cl100k_base`)
* **Word Count:** 53 words
* **Exact Raw Text:**
```text
PINCH; It is no shame: the fellow finds his vein, 
And yielding to him humours well his frenzy. ANTIPHOLUS OF EPHESUS; Thou hast suborn'd the goldsmith to arrest me. ADRIANA; Alas, I sent you money to redeem you, 
By Dromio here, who came in haste for it. DROMIO OF EPHESUS; Money by me!
```

#### 2. Local Context
* **Previous Chunk (`C146`):**
  > *"ANTIPHOLUS OF EPHESUS; Dined at home! Thou villain, what sayest thou?... ADRIANA; Is't good to soothe him in these contraries?"*
* **Target Chunk (`C147`):** As displayed above.
* **Next Chunk (`C148`):**
  > *"heart and goodwill you might; / But surely master, not a rag of money. ANTIPHOLUS OF EPHESUS; Went'st not thou to her for a purse of ducats? 60"*

#### 3. Structural Information
* **Act & Scene:** Act 4, Scene 4
* **Location:** A public place, Ephesus
* **Breadcrumb Path:** `ACT 4 \n43`
* **Parent ID:** `ACT 4 \n43\nSCENE 1  \n \nA public place...`
* **Speakers:** Pinch, Antipholus of Ephesus, Adriana, Dromio of Ephesus
* **Characters Mentioned:** Antipholus of Ephesus, Adriana, Dromio of Ephesus, "goldsmith"
* **Semantic/Document Type:** Dramatic Dialogue

#### 4. Gold Relationship
* **Benchmark Question:** `Q06` — *"Why is Antipholus of Ephesus arrested, and what role does Angelo the goldsmith play in the debt dispute?"*
* **Gold Role:** `SUPPORTING`
* **Required Claim Unit:** `Q06_C2` — *"Ephesus twin denied chain, arrested for 200 ducats debt."*
* **Annotation Rationale:** Contains Antipholus of Ephesus's explicit accusation: `Thou hast suborn'd the goldsmith to arrest me`, linking the goldsmith directly to his arrest.

#### 5. Retrieval Trace
* **Dense Q0 Cosine Rank (All 203 Chunks):** **#6** (Cosine Similarity: `0.5842` — Outstanding dense alignment for Q0!)
* **Q2 Subquery Dense Ranks:**
  - Subquery 1 (*"Antipholus of Ephesus character background arrest debt"*): Rank **#20** (Sim: `0.5000`)
  - Subquery 2 (*"Angelo goldsmith relationship debt dispute legal claim"*): Rank **#12** (Sim: `0.2295`)
  - Subquery 3 (*"Arrest of Antipholus of Ephesus scene Angelo goldsmith demand payment"*): Rank **#5** (Sim: `0.5968`)
* **BM25 Rank (All 203 Chunks):** `N/A` (> len of active BM25 matches)
* **Q2 Candidate Union Entry:**
  - $N=20$ Candidate Pool: **NO** (Pushed out of candidate union by competing subquery results in baseline deduplication)
  - $N=50$ Candidate Pool: **NO** (Base Q2 union); **YES** at $N=50$ in Phase 8A/8C (Rank #44)
  - $N=100$ Candidate Pool: **YES**
* **Cross-Encoder Direct Score:** **`+1.3017`** (Exceptionally high Cross-Encoder score!).

#### 6. Query-to-Chunk Relationship
* **Explicit Entity Overlap:** `ANTIPHOLUS OF EPHESUS`, `ADRIANA`, `DROMIO OF EPHESUS`, `goldsmith`.
* **Lexical Overlap:** `goldsmith`, `arrest`, `money`, `redeem`.
* **Terminology Overlap:** High. Contains exact words `goldsmith` and `arrest`.
* **Semantic Relationship:** Explicit dialogue regarding the arrest and goldsmith subornation.
* **Answer Concept Stated:** Explicit (`suborn'd the goldsmith to arrest me`).
* **Reliance on Neighboring Text:** Low; standalone accusation.
* **Indirect References:** None.
* **Verse/Archaic Mismatch:** Low.

#### 7. Context Dependency Classification
* **Primary Category:** `SELF_CONTAINED`
* **Secondary Category:** `CONTEXT_DEPENDENT`
* **Evidence:** Contains explicit statement `Thou hast suborn'd the goldsmith to arrest me`. Self-contained statement of arrest.

#### 8. Retrieval Failure Classification
* **Primary Mechanism:** `CANDIDATE_GENERATION_FAILURE` (Subquery Competition / Candidate Union Truncation)
* **Secondary Mechanism:** `RANKING_FAILURE` (At $N=50$)
* **Empirical Evidence:** Dense Q0 rank is **#6** and Cross-Encoder score is **`+1.3017`** (very strong). However, multi-query expansion subqueries generated 3 separate top-20 pools that filled the candidate union with long scene descriptions (`C105`, `C107`, `C155`), pushing short 287-char `C147` outside the top-20 candidate union.

---

## Required Summary Comparison Table

| Chunk | Question | Gold Role | Self/Context Status | Semantic Rank (Q0) | Q2 N20 | Q2 N50 | CE Rank | Primary Failure Mechanism | Empirical Evidence |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **`C15`** | `Q03` | `PRIMARY` | `ENTITY_DISCONNECTED` | **#142** | NO | NO | N/A | `REPRESENTATION_MISMATCH` | Dense rank #142 (sim 0.31); CE score -4.95; poetic metaphor ("drop of water") disconnected from QA keywords. |
| **`C29`** | `Q03` | `SUPPORTING` | `CONTEXT_DEPENDENT` | **#60** | NO | NO | N/A | `CHUNK_CONTEXT_PROBLEM` | 449-char dialogue fragment in 10-chunk scene; lacks entity names ("your man", "tardy master"); dense rank #60. |
| **`C42`** | `Q05` | `PRIMARY` | `LINGUISTICALLY_DIFFICULT` | **#99** | NO | NO | N/A | `REPRESENTATION_MISMATCH` | CE score positive (+0.58), but dense rank #99 (sim 0.35); 16th-century jargon ("sconce") blocks candidate entry. |
| **`C61`** | `Q03` | `SUPPORTING` | `SELF_CONTAINED` | **#33** | **YES** | **YES** | **#14** | `RANKING_FAILURE` | **Entered candidate pool via BM25 rank #7**; CE scored it -0.82 and demoted it to #14 (missed Top-10). |
| **`C147`** | `Q06` | `SUPPORTING` | `SELF_CONTAINED` | **#6** | NO | NO (Base) | N/A (#15 at N50) | `CANDIDATE_GENERATION_FAILURE` | Outstanding Q0 dense rank (#6) and CE score (+1.30); subquery competition/deduplication pushed it past N20 limit. |

---

## Cross-Case Analysis

### Answers to 10 Mandatory Diagnostic Questions

1. **Are the failures concentrated in dialogue/verse?**  
   *Yes.* All 5 target chunks (`C15`, `C29`, `C42`, `C61`, `C147`) are dramatic dialogue or verse soliloquies from *The Comedy of Errors*. Standard prose RAG retrievers struggle with non-prose theatrical formatting.

2. **Are the chunks semantically self-contained?**  
   *Mixed.* `C61` and `C147` are fully self-contained statements (`"Husband, I'll dine above..."`, `"suborn'd the goldsmith to arrest me"`). `C15` and `C42` are self-contained in meaning but use metaphor/slang. `C29` is heavily context-dependent.

3. **Do they lack explicit entity names?**  
   *Yes, for `C15` and `C29`.* `C15` refers to `"a mother and a brother"` instead of Adriana/Antipholus. `C29` refers to `"your man"` and `"tardy master"`. `C42`, `C61`, and `C147` contain explicit character names.

4. **Do they depend on neighboring context?**  
   *`C29` strongly depends on neighboring context (`C24`–`C28`, `C30`).* The remaining 4 chunks contain localized meaning but suffer from granularity or vocabulary issues.

5. **Do they contain archaic or indirect language?**  
   *Yes.* `C15` uses Renaissance poetic soliloquy verse ("drop of water", "almanac of my true date"). `C42` uses 16th-century slang ("sconce", "make a common"). `C29` uses stichomythic dialogue puns ("two hands", "mind upon mine ear").

6. **Are they absent from semantic retrieval, Q2 candidate generation, or only lost during CE ranking?**  
   - `C15` and `C42` are absent from semantic retrieval (dense ranks #142 and #99).
   - `C29` and `C147` are absent from Q2 candidate union at $N=20$ (dense ranks #60 and #6 with subquery dilution).
   - `C61` is **present in candidate generation** (BM25 rank #7), but **lost during CE reranking** (CE rank #14).

7. **Is there evidence that the current semantic representation is insufficient?**  
   *Yes, strong evidence.* `all-MiniLM-L6-v2` dense embeddings map `C15` (poetic soliloquy) to rank **#142** and `C42` (slang beating) to rank **#99**. Modern 384-dim dense models trained on web QA pairs perform poorly on 16th-century poetic verse.

8. **Is there evidence that chunk boundaries themselves are responsible?**  
   *Yes, for `C29`.* Slicing Act 2 Scene 1 into 10 separate 400-character micro-chunks fragments the dialogue, separating pronoun references from initial scene speaker setups.

9. **Is there evidence that the problem is primarily embedding-model related?**  
   *Yes, for `C15` and `C42`.* `C42` receives a positive Cross-Encoder score (`+0.5802`), proving that a cross-attention reranker recognizes its relevance, but the bi-encoder embedding model fails to surface it into the candidate pool.

10. **Which hypothesis is currently best supported, and what evidence supports it?**  
    *Supported Multi-Factor Hypothesis:*  
    - **Hypothesis A (Embedding Domain Shift):** Bi-encoder embeddings (`all-MiniLM-L6-v2`) fail on poetic/metaphoric verse (`C15`, `C42`). Supported by dense ranks #142 and #99 despite positive CE score for `C42`.  
    - **Hypothesis B (Dialogue Chunk Fragmentation):** Micro-chunking (300–500 chars) breaks dialogue context (`C29`). Supported by 10-chunk fragmentation in Scene 1.  
    - **Hypothesis C (Reranker Score Calibration):** Cross-Encoder penalizes short verse dialogue chunks (`C61`, CE rank #14 vs BM25 rank #7).

---

## Methodological Distinction

To maintain strict diagnostic rigor, observed empirical facts are explicitly isolated from interpretations and hypotheses:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                     OBSERVED FACT                                      │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. C15 dense rank across 203 chunks is #142 (sim 0.3124); CE score is -4.9510.         │
│ 2. C42 dense rank across 203 chunks is #99 (sim 0.3512); CE score is +0.5802.          │
│ 3. C61 BM25 rank is #7; enters N=20 candidate pool; CE rank is #14 (score -0.8227).    │
│ 4. C147 Q0 dense rank is #6; CE score is +1.3017; absent from N=20 Q2 candidate union. │
│ 5. C29 is a 449-char chunk in a 10-chunk scene; dense rank #60; BM25 N/A.               │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                     INTERPRETATION                                     │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. C15 fails due to bi-encoder & cross-encoder inability to match poetic metaphors.   │
│ 2. C42 candidate failure is driven by bi-encoder embedding gap (CE score is positive). │
│ 3. C61 fails strictly at the reranking selection stage (candidate generation succeeded).│
│ 4. C147 candidate failure is caused by multi-query subquery competition at N=20.       │
│ 5. C29 candidate failure is driven by micro-chunking dialogue fragmentation.           │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                       HYPOTHESIS                                       │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. Domain-adapted or higher-dimensional dense embeddings will resolve C15 & C42.      │
│ 2. Parent-child context aggregation will resolve C29 dialogue fragmentation.           │
│ 3. Adjusting reranker score calibration or context window (Top-15) will resolve C61.   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Output Artifacts & Verification

* **Diagnostic JSON Artifact:** Saved at [`evaluation/results/phase9a_autopsy_results.json`](file:///d:/Fine%20Tunning/ADVANCE%20RAG/RAG/evaluation/results/phase9a_autopsy_results.json) containing full character counts, token counts, exact raw text, local context, structural info, subquery ranks, and depth analysis for all target chunks.
* **Diagnostic Report File:** Saved at [`evaluation/reports/phase9a_failed_evidence_autopsy.md`](file:///d:/Fine%20Tunning/ADVANCE%20RAG/RAG/evaluation/reports/phase9a_failed_evidence_autopsy.md).
