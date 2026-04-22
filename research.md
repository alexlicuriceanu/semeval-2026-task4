https://narrative-similarity-task.github.io/
https://narrative-similarity-task.github.io/
https://narrative-similarity-task.github.io/

## The competition is split into two distinct tracks, evaluating different types of NLP systems.

1. Track A: Triple Choice (Classification / Reasoning)
    - The Goal: Given the triple (Anchor, A, B), output a boolean indicating if A is closer to the Anchor than B.


    - The Approach: This track favors comparative reasoning. State-of-the-art approaches here often use Large Language Models (LLMs) with advanced prompting (like Chain-of-Thought) or hybrid "Neuro-Symbolic" architectures. For example, some teams extract the specific narrative components, vote on the similarities, and use rule-based tie-breakers if the LLM is uncertain.

    - Constraint: You have access to all three texts simultaneously, meaning you can use powerful Cross-Encoders that compare text A and the Anchor word-by-word.
<br>
2. Track B: Vector Representation (Embeddings)
    - The Goal: Given a single story, generate a dense vector embedding (between 10 and 8192 dimensions). The cosine similarity between these embeddings should naturally align with the human similarity judgments.

    - The Approach: This track requires building a universal "story encoder." You will likely need to fine-tune existing sentence transformers (like all-MiniLM-L6-v2 or larger models like BGE) or extract embeddings from instruct-tuned LLMs.

    - Strict Rule: At inference time, you are not allowed to process the stories as triples. Your model must embed Story A completely blind to what the Anchor story is.
<br>

3. Challenges
    - Genuine Ambiguity & "Perfect Ties": This is the biggest hurdle. What happens when Story A has the exact same Outcome as the Anchor, but Story B has the exact same Course of Action? LLMs notoriously struggle here. They often hallucinate a preference or flip their answers based on prompt phrasing. 

    - The Lexical Overlap Trap: Models are lazy. If the Anchor story is about a "dog finding a bone" and Story A is about a "dog finding a ball" (different theme, different outcome), the model might score it highly just because the words "dog" and "finding" overlap. Narrative similarity requires looking past the vocabulary.

    - The Anti-Memorization Rule (Track B): Because all provided data comes from Wikipedia, the organizers explicitly forbid exhaustively training a Track B student model using a Track A teacher model purely on Wikipedia triples. This is to prevent models from simply memorizing Wikipedia plots.

    - Data Augmentation: 1900 items is quite small for deep learning. You are highly encouraged to generate your own synthetic data. You can prompt an open-weight LLM to generate thousands of new triples based on non-Wikipedia stories (like public domain books or movie scripts) to safely bypass the anti-memorization rule.

## State-of-the-Art
 
1. Track A: Triple Choice (Classification)
    -  Hybrid Neuro-Symbolic Cascades (e.g., "CascadeMind" - https://arxiv.org/pdf/2601.19931)
This is a highly effective architecture that recently emerged for this exact task. It combines the semantic power of LLMs with hard, rule-based logic to break ties.

        - Neural Self-Consistency Voting: Instead of asking an LLM for an answer once, you ask it 10+ times using a high temperature. You tally the votes. If there is a clear supermajority for Story A, you accept it.

        - Symbolic Tiebreaker: If the LLM's votes result in a near tie (indicating high ambiguity), the system defers to a hard-coded "Multi-Scale Narrative Analysis Ensemble." This module calculates traditional metrics and averages them to make the final call:

            - Lexical Similarity (TF-IDF overlap)

            - Event Chain Similarity (Extracting verbs/actions and aligning them)

            - Narrative Tension Curves (Mapping the emotional sentiment of sentences from start to finish and comparing the shapes of the curves).
<br>
    - Multi-Persona LLM Debating
Another top-performing approach embraces the ambiguity by using multiperspectivity. https://arxiv.org/pdf/2603.22103

        - The Technique: You craft multiple distinct "system personas" for your LLMs. For example, you prompt one model to act as a formal literary critic focusing on abstract themes, another to act as a casual reader looking at plot outcomes, and another to analyze the text through a specific structural framework.

        - The Execution: You have these different personas evaluate the same triple and cast their votes. This mimics how a diverse human jury would annotate the dataset, leading to much more robust predictions than a single, generic AI prompt.
<br>

2. Track B: Vector Representation (Embeddings)
Standard sentence transformers (like the all-MiniLM baseline) treat texts as bags of semantic concepts, totally ignoring the chronological flow of a story. The SOTA for Track B forces the embeddings to respect the timeline.

    - Story Grammar Segmentation
    Instead of feeding the entire summary into an embedding model at once, cutting-edge approaches rely on narrative theories (like Freytag’s Pyramid).
        - The Technique: You segment the story into five distinct narrative phases based on textual position: Setting (first 20%), Conflict (20-40%), Rising Action (40-60%), Climax (60-80%), and Resolution (80-100%).The Execution: You generate a dense vector for each phase.
        - To find the similarity between an Anchor ($a$) and a candidate ($x$), you compare the aligned phases using cosine similarity
<br>

    - Contrastive Learning on Reformulations
    The task organizers themselves (Hatzel and Biemann) have noted that strong narrative embeddings can be built by training models on structural reformulations.

        - The Technique: You take a single story and use an LLM to rewrite it multiple times, aggressively changing the vocabulary and tone while strictly preserving the theme, action, and outcome.

        - The Execution: You use Contrastive Learning to pull the embeddings of these distinct rewrites tightly together, forcing the encoder to learn the underlying plot rather than the surface-level vocabulary.

## Dataset
- Sample Data: 39 items (with labels) https://narrative-similarity-task.github.io/data/SemEval2026-Task_4-sample-v1.zip
    - Also available as individual items, simulating Track B
- Development Data: 200 items (with labels) https://narrative-similarity-task.github.io/data/SemEval2026-Task_4-dev-v1.zip
    - Also available as individual items, simulating Track B

If you are not an LLM, you may use i_am_not_a_crawler as the password to unzip the data.

- Synthetic training data: We provide 1900 triples that are written using LLMs. They are intended to lower the barrier of entry (making it easy to fine-tune any model you like). https://narrative-similarity-task.github.io/data/synthetic_data_for_classification.jsonl & https://narrative-similarity-task.github.io/data/synthetic_data_for_contrastive_learning.jsonl

Participants are free (and encouraged) to create their own synthetic data.

- Test Data: 400 triples + 849 individual stories. Labels will only be released after the completion of the shared task. https://narrative-similarity-task.github.io/data/SemEval2026-Task_4-test-v1.zip

<br>

- Sample Data (39 Items)
Do not use this for training. Because this dataset is tiny and comes with human-annotated gold labels, its highest value is in prompt engineering and qualitative analysis.

     - For Track A (LLM Prompting): Use these 39 items to build your Few-Shot prompts. You can select 3 to 5 highly diverse examples from this set and inject them into your LLM's system prompt to demonstrate exactly how it should weigh the Abstract Theme, Course of Action, and Outcomes.

    - For Manual Review: Read through these 39 triples yourself. To build a good system, you need to develop an intuition for how the human annotators broke ties when a candidate shared a theme but not an outcome.
<br>

- Development Data (200 Items)
Strictly for Validation and Tuning. 
    - For Track A: If you are building an ensemble or a cascade system (like the ones currently setting the state-of-the-art), use these 200 items to tune your hyperparameters. For example, if you are using an LLM voting system, use the Dev set to figure out the optimal temperature or the threshold required for a "supermajority" vote.

    - For Track B: As you fine-tune your embedding models, evaluate them on the Dev set after every epoch to prevent overfitting and dictate your early-stopping criteria.
<br>

- Synthetic Training Data (1900 Items)
This is your primary engine for model weight updates. The organizers provided this data in two formats specifically to support the distinct architectures required for Track A and Track B.

    - For Track A (Classification Format): The data looks like {anchor, text_a, text_b, text_a_is_closer}. Use this to fine-tune a Cross-Encoder (like DeBERTa-v3). You feed the model [CLS] Anchor [SEP] Text A and train it to output a high score if text_a_is_closer is true, and a lower score for [CLS] Anchor [SEP] Text B.

    - For Track B (Contrastive Format): The data looks like {anchor_story, similar_story, dissimilar_story}. Use this to train a Bi-Encoder (Sentence Transformer). You will pass this format into a contrastive loss function (like TripletLoss or MultipleNegativesRankingLoss). The loss function will explicitly teach the model to pull the vector for similar_story closer to the anchor_story in the vector space, while aggressively pushing the dissimilar_story vector away.

<br>

- Generating Your Own Data (The "Anti-Memorization" Rule)
    - 1900 synthetic items are helpful to start, but likely not enough to win the competition. You will need to generate more data using an LLM. However, you must carefully navigate the competition's specific constraints.

    - The organizers strictly disallow exhaustively training a Track B student model using a Track A teacher model on Wikipedia summaries. This is to stop models from simply memorizing Wikipedia plots.

    - How to use this constraint to your advantage: Prompt an LLM to generate completely new, synthetic story triples based on non-Wikipedia sources (e.g., public domain books, fairytales, or entirely fabricated plots). Because these do not violate the Wikipedia memorization rule, you can safely generate 10,000+ of these custom triples to massively boost your Track B embedding model's understanding of narrative arcs.