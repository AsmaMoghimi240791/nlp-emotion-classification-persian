**Model Card: ParsBERT (Emotion Classification in Persian)**  


---

### Model Details

### Model Overview ###

Architecture

ParsBERT is a transformer-based monolingual language model developed specifically for the Persian language. It follows the BERT-base architecture, consisting of 12 transformer encoder layers, 12 self-attention heads, and a hidden size of 768, totaling approximately 110 million parameters. The model was originally trained on a large Persian corpus to learn the contextual representation of words—effectively capturing the positive or negative weight of each token within sentences.
To adapt it for emotion recognition, we fine-tuned the pre-trained ParsBERT model on a labeled dataset covering seven emotional categories: happiness, sadness, fear, anger, surprise, disgust, and neutral. This was achieved by adding a dense layer with softmax activation to output emotion class probabilities.

Purpose

The primary goal of this project was to develop an effective emotion detection system for Persian text, a domain with relatively few dedicated NLP tools. Given that multilingual models like mBERT tend to perform poorly on low-resource languages such as Persian, we chose ParsBERT—a monolingual model—to better capture the nuances of Farsi, especially due to the presence of shared words across regional dialects with different meanings.
Emotion recognition in text has a wide range of modern applications. For instance, platforms like Instagram and Google use similar models to tailor content, ads, or recommendations based on user mood or emotional tone. This capability can also enhance targeted advertising by identifying the best moments—such as when users are happy or sad—to display emotionally resonant ads.

Development context

Our development assumed that monolingual pretraining provides more accurate language representations in low-resource settings. While the ParsBERT model was already trained on a vast corpus, it had not been tailored for emotion classification. Thus, we fine-tuned it using a custom Persian dataset annotated with emotional categories.
We initially began training with a small dataset, but due to poor performance, we incrementally increased the dataset size to improve the accuracy and F1-score. This iterative data expansion approach helped enhance model performance over time. The project also faced typical constraints such as limited annotated data, computational resources, and time, yet still achieved promising results for a seven-class classification task in a low-resource language setting.

### Intended Use ###

This model is designed to detect emotions in Persian-language text and can be used in several real-life applications. One of the main areas where it can be helpful is in social media platforms like Instagram or in search engines like Google. When users post or search for something, the emotional tone behind their text can be picked up by the model, and that emotion can then be used to recommend related content or advertisements. For example, if someone is feeling happy, the system could suggest cheerful ads or videos, and if someone shows signs of sadness or anger, it could adjust what content is shown based on that mood. This can help improve how personalized and effective ads are, since certain advertisements work better when shown at specific emotional moments.
Another practical use is in analyzing user feedback, comments, or even transcripts from interviews or customer service chats. By understanding how people feel through their words, companies and media platforms can better respond to users or understand public opinion. This is especially useful in marketing, media analysis, or customer support.
However, there are some limitations. The model is trained only on Persian, so it won’t perform well on content written in other languages. It’s also not meant to be used for serious decisions in fields like healthcare, law, or mental health without human supervision, because misinterpreting someone’s emotion could have serious consequences. Despite these limitations, the model is very useful for improving emotional understanding in Persian texts, especially since there aren’t many tools like this available for low-resource languages like Farsi. It supports better content recommendations, more targeted advertising, and a deeper analysis of how users feel — which can be really valuable for companies and platforms working in Persian-language environments.
---

### Model Type
Sequence classification transformer model based on **ParsBERT** (a BERT variant pre-trained on Persian text).

### Model Version
Fine-tuned variant for 7-class Persian emotion classification using HuggingFace `AutoModelForSequenceClassification`.


### Dataset Details

For training our model, we mainly used data collected from Twitter (now called X), which was a great source because it included a wide range of writing styles—formal, informal, and semi-formal. This variety helped the model learn how emotions are expressed in different ways. To make the dataset more diverse, we also added transcripts from videos, which gave us access to spoken language converted into text. On top of that, we generated synthetic sentences for each of the seven emotions to make sure the model had balanced examples for all emotional categories like happiness, sadness, anger, and so on.
At first, we tried extracting features like part-of-speech (POS) tags and tokenized forms of the sentences, thinking that these might help the model better understand sentence structure and improve emotion detection. But in practice, these extra features added noise and actually reduced performance. So, we decided to continue training only on the raw text, which gave us the best results.
Since Persian shares a lot of vocabulary with other languages spoken in nearby countries, we used a monolingual language model instead of a multilingual one. Multilingual models sometimes confuse the meanings of shared words when they appear in different contexts, which can lead to incorrect emotional interpretations. By using a monolingual Persian model, we avoided that issue and achieved more accurate results. However, our dataset is focused only on Persian, so it doesn't represent other languages or cultures, and the model may not perform well on non-Persian texts or mixed-language input. We also had limited access to annotated data, so while we tried to make our dataset diverse, it may not cover all dialects or regions equally.
---

### Performance Metrics and Evaluation 

Our ParsBERT-based model achieved an overall accuracy of 56.65% across the seven emotion classes. While it performed well in detecting strong emotions like anger and sadness, it struggled with subtler or overlapping categories such as fear, disgust, and especially neutral, which often served as a catch-all category for ambiguous inputs. The confusion matrix indicated overprediction of anger, and vague or short sentences were commonly mispredicted. Misclassifications largely arose from lexical ambiguity, cultural expressions, and lack of enough emotional cues in short texts. These findings highlight the need for better contextual understanding and class balancing in future iterations.
link==>

### Explainability and Transparency

In this project we tested three explainability methods on ParsBERT. The Gradient ×
Input method was useful to quickly see which tokens mattered, but the results were
often noisy because even neutral words got high scores. The LRP method worked
much better, since it highlighted the emotional words more clearly and closer to human
intuition, while ignoring irrelevant tokens. Input perturbation gave us another view by
testing how the model’s confidence changes when removing tokens. In some cases,
confidence dropped sharply when key emotional words were removed, showing the
model’s reliance on them, while in other cases the confidence decreased slowly,
meaning the model spread importance across more tokens. Overall, we can say thatLRP gave the most understandable explanations, perturbation showed how stable the
model is, and Gradient × Input, even if noisy, still helped us to get a first impression.
Using them together gave us a full picture of how ParsBERT makes its predictions.

link ==>

### Recommendations for Use
This model can be used in real-world situations where understanding emotion from text is important. One useful application is in marketing, especially for companies that work with video content. For example, if a company has videos of customer interviews or product reviews, the model can transcribe the audio into text and then analyze the emotional tone of each sentence. This can help businesses understand how their audience feels and target their ads more effectively, without having to manually search through content like on Instagram or other social platforms. It helps brands connect with the right audience at the right emotional moment, which can be very valuable for improving engagement and sales.
However, there are some important things to keep in mind when using the model. First, the quality of the input matters a lot—if the audio is unclear or the transcription is inaccurate, the emotion prediction might not be reliable. Also, while the model is powerful, it should be used responsibly. Since it can pick up on emotional signals from text, there’s a risk of invading user privacy if it's used without consent or in sensitive situations. Users should be aware that using AI this way could raise ethical questions, especially if it starts to feel like people are being constantly monitored or profiled. For that reason, the model is best used in controlled environments with clear user consent, and not in places where privacy or safety might be compromised.
Overall, this model is a useful tool for media companies, marketers, or content analysts who want to better understand emotional responses in Persian-language media. It works best when combined with high-quality input and ethical usage policies.

---

### Sustainability Considerations
We trained our model using the NVIDIA L40S GPU, which is a high-performance, server-grade card with a power draw of around 350 watts. Since we only fine-tuned an existing model (ParsBERT) and limited training to one hour, the total energy consumption was relatively low—about 0.35 kWh. Based on average energy sources in Europe, this translates to approximately 0.14 kg of CO₂ emissions.
To reduce the environmental impact, we used a pre-trained model instead of training from scratch, which saves a significant amount of energy and compute resources. For future development or deployment, we recommend using lower-power GPUs for inference, implementing model compression techniques like quantization, and running training on servers powered by renewable energy whenever possible.


| **Metric**               | **Value (Est.)**                     |
| ------------------------ | ------------------------------------ |
| **GPU Used**             | NVIDIA L40S                          |
| **Training Time**        | 1 hour                               |
| **Power Consumption**    | ~0.35 kWh                            |
| **Estimated CO₂ Output** | ~0.14 kg CO₂ *(based on EU average)* |
| **Training Type**        | Fine-tuning a pre-trained model      |




### Contact and Repository
Model developed by Group 15, BUas AI & Data Science Year 2 (2025).  
Repository: [Link to group GitHub repository here]  
Error analysis: [Task 9 ](https://github.com/BredaUniversityADSAI/fae2-nlpr-group-group-15-1/tree/main/Error%20Analysis)  
XAI & Interpretability:[Task 10](https://github.com/BredaUniversityADSAI/fae2-nlpr-group-group-15-1/tree/main/XAI)

---

_This model card was prepared in alignment with the principles from "Model Cards for Model Reporting" (Mitchell et al., 2019)._