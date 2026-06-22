***Findings – Round Translation Analysis***
**Overview**

We analyzed 150 Persian-to-English translations produced by our fine-tuned mT5-small model.
Each sentence received an accuracy score (3 = correct, 2 = minor mistake, 1 = incorrect) and a short comment explaining the issue.

The goal was to see how well the model performs on real Persian sentences and to understand what types of errors it makes most often.
We found that while the model often produces fluent and grammatically correct English, it doesn’t always capture the real meaning.
Overall, it performs well on simple, factual, and neutral sentences, but struggles with idioms, figurative language, or flexible Persian grammar.

**1. Main Types of Mistakes**
*1. Semantic drift (confusion of meaning)*

This was the most common problem.
Many translations sounded fine in English but failed to communicate the actual meaning of the original Persian sentence.
The model tended to guess based on surface words instead of full sentence context.

Example:
“می‌تونیم روش حساب کنیم” (we can count on him) → “we can calculate it.”

*2. Hallucination or repetition*

Sometimes the model repeated words or added nonsense phrases not found in the source,
such as “going to be going to be” or random words like “sandwich” or “United States.”
This happened mostly with long or unclear sentences, a sign that the model lost direction or didn’t know how to finish.

*3. Confusing similar words (polysemy)*

Persian contains many words that look or sound alike but have different meanings depending on context.
The model often chose the most frequent meaning it had seen in training instead of the correct one for that sentence.

*4. Problems with tense and word order*

Persian allows flexible word order and often omits subjects because they’re understood from context.
English, on the other hand, requires a fixed order and explicit subjects.
The model sometimes forced English syntax in a way that changed the intended tone or meaning.

*5. Literal translation of idioms and phrases*

Persian idioms rarely have direct English equivalents, but the model translated them word-for-word.

Example:
“خوش می‌گذره” (we’re having fun) → “it’s very happy.”

This produces translations that are grammatically fine but sound unnatural or incorrect.

 **2. Why These Mistakes Happen**

Most of these issues make sense from both linguistic and technical perspectives.

Persian grammar is flexible, while English is strict, structure doesn’t transfer easily.

Persian meaning depends heavily on tone, context, and shared understanding, small models can’t always infer this.

Idioms and cultural phrases don’t appear often in general training data, so the model defaults to literal translation.

Polysemic words confuse the model when it only focuses on local context instead of the full sentence.

Model size and data limits, mT5-small doesn’t have enough capacity to capture deep semantic and cultural nuance.

**3. Patterns in Performance**

We noticed that translations improved toward the end of the dataset.
The early sentences were more conversational and emotional, while the later ones were factual and structured,
often about football, teamwork, or films — areas with clearer grammar and familiar vocabulary.

This shows that the model performs best on structured, topic-based sentences, similar to what it saw in its training data.
It struggles more with informal or emotional speech, which it hasn’t been widely exposed to.

**4. How to Improve the Model**

To improve Persian-to-English translation quality:

Use a larger model, such as mT5-base or mT5-large, which can handle context and idioms better.

Fine-tune on clean, context-rich datasets like TED2020, MIZAN, or Persian-English subtitles that include both formal and casual speech.

Train on longer sequences (paragraphs instead of isolated sentences) to provide richer context.

Diversify the dataset, mix conversational and formal data.

Add a post editing or filtering step to remove repetitive or nonsensical translations before scoring.

**5. Key Takeaways**

The model is grammatically strong but semantically weak, it builds proper sentences but misses deeper meaning.

Most errors come from misinterpreting context, not broken grammar.

Performs well on short, neutral, or direct sentences, but struggles with idioms, tone, and cultural meaning.

The problems are linguistically explainable, Persian’s structure and idiomatic nature are hard for smaller multilingual models.

Better data, larger models, and context aware training could significantly improve fluency and accuracy.

**Conclusion**

Our mT5-small model is a strong starting point for Persian-English translation.
It can handle basic and clear sentences, but it still lacks the deeper context awareness needed for natural, accurate translations.
The errors we found are not random, they reflect both the complexity of Persian and the model’s limited exposure to conversational data.

With better datasets and fine-tuning, it could reach much higher translation quality and produce results that feel more fluent and human like.
