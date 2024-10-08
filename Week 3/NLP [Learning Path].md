# NLP [Learning Path]

# Fundaments of Text Analysis [Module 1]

## Introduction

- **Natural Language Processing** (NLP): Area within AI that deals with understanding written or spoken language.
- Use case: Speech Recognition, Sentiment Analysis, Text Extraction, Summarization, Conversational AI, etc.

> **Azure AI Language** is a cloud-based service that includes features for understanding and analyzing text. Azure AI Language includes various features that support sentiment analysis, key phrase identification, text summarization, and conversational language understanding.
> 

## **Understand Text Analytics**

> *Corpus →* body of text
> 
- Earliest techniques used to analyze text with computers involve statistical analysis of a body of text (a *corpus*) to infer some kind of semantic meaning.
- If you can determine the most commonly used words in a given document, you can often get a good idea of what the document is about.

### Tokenization

- corpus is broken down into *tokens*
    - tokens can be generated from single words, partial words, or combinations of words and punctuation.

> Tokenization in different cases:
> 
> - **Text normalization**: We *normalize* the text by removing punctuation and changing all words to lower case.
> - **Stop word removal:** words that should be excluded from the analysis, 
> For example, "*the*", "*a*", or "*it*" because they add little semantic meaning.
> - **n-grams:**  are multi-term phrases, By considering words as groups, a machine learning model can make better sense of the text.
> - **Stemming:** technique in which algorithms are applied to consolidate words before tokenization, so that words with the same root, like "power", "powered", and "powerful", are interpreted as being the same token.

### Frequency analysis

- means to count the number of occurrences of each token.

> Simple Frequency Analysis
> 
> - can be an effective way to analyze a single document.
> - but for multiple documents within the same corpus, we have to determine which tokens are most relevant in each document.
> - ***Term frequency -** inverse document frequency* (TF-IDF) is a common technique in which a score is calculated based on how often a word or term appears in one document compared to its more general frequency across the entire collection of documents.

### Machine learning for text classification

- Another text analysis technique is to use a classification algorithm, such as logistic regression
- To train a machine learning model that classifies text based on a known set of categorizations.
- A common application of this technique is to train a model that classifies text as positive or negative in order to perform sentiment analysis or opinion mining.
- To an train a classification model using the tokenized text as *features* and the sentiment (0 or 1) a *label*. The model will encapsulate a relationship between tokens and sentiment.

### Semantic language models

![image.png](../Resorces/Images/Week%203/Semantic%20language%20models%20image.png)

- At the heart of these models is the encoding of language tokens as vectors (multi-valued arrays of numbers) known as *embeddings.*
- We think of the elements in a token embedding vector as coordinates in multidimensional space,
- Each token occupies a specific "location." The closer tokens are to one another along a particular dimension, the more semantically related they are.
- In other words, related words are grouped closer together
- The locations of the tokens in the embeddings space include some information about how closely the tokens are related to one another.

## Get started with text analysis

- Azure AI Language's text analysis features include (just for reference)
    - **Named entity recognition** identifies people, places, events, and more. This feature can also be customized to extract custom categories.
    - **Entity linking** identifies known entities together with a link to Wikipedia.
    - **Personal identifying information (PII) detection** identifies personally sensitive information, including personal health information (PHI).
    - **Language detection** identifies the language of the text and returns a language code such as "en" for English.
    - **Sentiment analysis and opinion mining** identifies whether text is positive or negative.
    - **Summarization** summarizes text by identifying the most important information.
    - **Key phrase extraction** lists the main concepts from unstructured text.

### Entity recognition and linking

- It return a list of *entities* in the text that it recognizes.
- An entity is an item of a particular type or a category;
- and in some cases, subtype, such as those as shown in the following table.
- Table of entities identified
    
    
    | Type | SubType | Example |
    | --- | --- | --- |
    | Person |  | "Bill Gates", "John" |
    | Location |  | "Paris", "New York" |
    | Organization |  | "Microsoft" |
    | Quantity | Number | "6" or "six" |
    | Quantity | Percentage | "25%" or "fifty percent" |
    | Quantity | Ordinal | "1st" or "first" |
    | Quantity | Age | "90 day old" or "30 years old" |
    | Quantity | Currency | "10.99" |
    | Quantity | Dimension | "10 miles", "40 cm" |
    | Quantity | Temperature | "45 degrees" |
    | DateTime |  | "6:30PM February 4, 2012" |
    | DateTime | Date | "May 2nd, 2017" or "05/02/2017" |
    | DateTime | Time | "8am" or "8:00" |
    | DateTime | DateRange | "May 2nd to May 5th" |
    | DateTime | TimeRange | "6pm to 7pm" |
    | DateTime | Duration | "1 minute and 45 seconds" |
    | DateTime | Set | "every Tuesday" |
    | URL |  | "`https://www.bing.com`" |
    | Email |  | "`support@microsoft.com`" |
    | US-based Phone Number |  | "(312) 555-0176" |
    | IP Address |  | "10.0.1.125" |

### **Language detection**

- You can submit multiple documents at a time for analysis.
- For each document submitted the service will detect:
    - The language name (for example "English").
    - The ISO 639-1 language code (for example, "en").
    - A score indicating a level of confidence in the language detection.

Example: 

> **Review 1**: "*A fantastic place for lunch. The soup was delicious.*"
> 
> 
> **Review 2**: "*Comida maravillosa y gran servicio.*"
> 
> **Review 3**: "*The croque monsieur avec frites was terrific. Bon appetit!*"
> 

| Document | Language Name | ISO 6391 Code | Score |
| --- | --- | --- | --- |
| Review 1 | English | en | 1.0 |
| Review 2 | Spanish | es | 1.0 |
| Review 3 | English | en | 0.9 |
- Notice that the language detected for review 3 is English, despite it containing a mix of English and French.
- The language detection service will focus on the ***predominant* language** in the text.
- The confidence score might be less than 1 as a result of the mixed language text.

### **Sentiment analysis and opinion mining**

- This capability is useful for detecting positive and negative sentiment in social media, customer reviews, discussion forums and more.
- The service returns sentiment scores in three categories: positive, neutral, and negative. In each of the categories, a score between 0 and 1 is provided.
- Scores indicate how likely the provided text is a particular sentiment.

### **Key Phrase Extraction**

- Key phrase extraction identifies the main points from text.
- Can be used to summarize the main points.

### **Create a resource for Azure AI Language**

> 🔵 In Azure AI - Language Studio

---

# Fundamentals of question answering with the Language Service  [Module 2]

## **Introduction**

- We want personal responses to our queries, without having to read in-depth documentation to find answers.
- Conversational AI describes solutions that enable a dialog between an AI agent and a human.

## **Understand question answering**

- Question answering is used to build bot applications that respond to customer queries.
- Question answering applications provide a friendly way for people to get answers to their questions and allows people to deal with queries at a time that suits them, rather than during office hours.
- he user gets an answer to their question quickly, and only gets passed to a person if their query is more complicated.

## **Get started with custom question answering**

## **Exercise - Use question answering with Language Studio**

> 🔵 In Azure AI - Language Studio

---

# Fundamentals of conversational language understanding  [Module 3]

## Introduction

- In 1950, British mathematician, Alan Turing devised the *Turing Test*  (aka *Imitation Game)*  and hypothesizes that if a dialog is natural enough, you might not know whether you're conversing with a human or a computer.
- To pass the test, computers not only need to be able to accept language as input (either in text or audio format), but also to be able to interpret the semantic meaning of the input
- i.e. ***Understand*** what is being said.
- **CLU -** Conversational Language Understanding

### **Describe conversational language understanding**

3 Major concepts to understand CLU:

- **Utterances**
    - A command by a user that your application must interpret.
    - e.g: Turn on the lights.
- **Entities**
    - An entity is an item to which an utterance refers.
    - e.g: Turn on the lights.
- **Intents**
    - An intent represents the purpose, or goal, expressed in a user's utterance.
    - A CLU application defines a model consisting of intents and entities.
    - Utterances are used to train the model to identify the most likely intent and the entities to which it should be applied based on a given input.
    - e.g: Turn on the lights.

## **Get started with conversational language understanding in Azure**

> To use CLU capabilities in Azure:
> 
> - **Azure AI Language**: A resource that enables you to build apps with industry-leading natural language understanding capabilities without machine learning expertise. You can use a language resource for *authoring* and *prediction*.
> - **Azure AI services**: A general resource that includes CLU along with many other Azure AI services. You can only use this type of resource for *prediction*.

### **Authoring (To create an authoring resource)**

- To train a model, start by defining the entities and intents that your application will predict as well as utterances for each intent that can be used to train the predictive model.

### **Training the model**

- Training is the process of using your sample utterances to teach your model to match natural language expressions that a user might say to probable intents and entities.
- Training and testing is an iterative process.
- After you train your model, you test it with sample utterances to see if the intents and entities are recognized correctly.
- If they're not, make updates, retrain, and test again.

### **Predicting**

- When you are satisfied with the results from the training and testing, you can publish your CLU application to a prediction resource for consumption.
- Client applications can use the model by connecting to the endpoint for the prediction resource, specifying the appropriate authentication key; and submit user input to get predicted intents and entities. The predictions are returned to the client application, which can then take appropriate action based on the predicted intent.

## **Exercise - Use Conversational Language Understanding with Language Studio**

> 🔵 In Azure AI - Language Studio

---

# Fundamentals of Azure AI Speech [Module 4]

## Introduction

- AI speech capabilities enable us to manage system with voice instructions, get answers from computers for spoken questions, generate captions from audio, and much more.
- the AI system must support two capabilities:
    - **Speech recognition** - the ability to detect and interpret spoken input
    - **Speech synthesis** - the ability to generate spoken output

## **Understand speech recognition and synthesis**

### **Speech recognition**

- **Speech recognition** takes the spoken word and converts it into data that can be processed - often by transcribing it into text.
- Speech patterns are analyzed in the audio to determine recognizable patterns that are mapped to words.
- To accomplish this, multiple models are used like :
    - An *acoustic* model that converts the audio signal into phonemes (representations of specific sounds).
    - A *language* model that maps phonemes to words, usually using a statistical algorithm that predicts the most probable sequence of words based on the phonemes.
- Example use case of Speech recognition:
    - Providing closed captions for recorded or live videos
    - Creating a transcript of a phone call or meeting
    - Automated note dictation
    - Determining intended user input for further processing

### **Speech synthesis**

- **Speech synthesis** is concerned with vocalizing data, usually by converting text to speech.
- To synthesize speech, the system typically ***tokenizes*** the text to break it down into individual words, and assigns phonetic sounds to each word.
- It then breaks the phonetic transcription into *prosodic* units (such as phrases, clauses, or sentences) to create phonemes that will be converted to audio format.
- These phonemes are then synthesized as audio and can be assigned a particular voice, speaking rate, pitch, and volume.
- Example use case of speech synthesis:
    - Generating spoken responses to user input
    - Creating voice menus for telephone systems
    - Reading email or text messages aloud in hands-free scenarios
    - Broadcasting announcements in public locations, such as railway stations or airports

## **Get started with speech on Azure**

- Azure offers both speech recognition and speech synthesis capabilities through **Azure AI Speech** service.
    - The **Speech to text** API
    - The **Text to speech** (TTS) API

## **The Speech to text API**

- You can use Azure AI Speech to text API to perform real-time or batch transcription of audio into a text format.
- The model is optimized for two scenarios, conversational and dictation.
- You can also create and train your own custom models including acoustics, language, and pronunciation if the pre-built models from Microsoft do not provide what you need.

### Real-time transcription

- Real-time speech to text allows you to transcribe text in audio streams.
- Your application streams (transmits) the audio to the (Azure) service, which returns the transcribed text.

### **Batch transcription**

- Not all speech to text scenarios are real time. You might have audio recordings stored on a file share, a remote server, or even on Azure storage.
- Batch transcription should be run in an asynchronous manner because the batch jobs are scheduled on a *best-effort basis*.
- Normally a job will start executing within minutes of the request but there is no estimate for when a job changes into the running state.

## The Text to Speech (TTS) API

- The text to speech API enables you to convert text input to audible speech.

### **Speech synthesis voices**

- When you use the text to speech API, you can specify the voice to be used to vocalize the text.
- This capability offers you the flexibility to personalize your speech synthesis solution and give it a specific character.
- The service includes multiple pre-defined voices with support for multiple languages and regional pronunciation.
- You can also develop custom voices and use them with the text to speech API

## **Exercise - Explore Speech Studio**

> 🔵 In Azure AI - Language Studio

---

# Fundamentals of language translation [Module 5]

## Understand translation concepts

- One of the many challenges of translation between languages is that words don't have a one-to-one replacement between languages.

> **Literal and semantic translation**
> 
> - Early attempts at machine translation applied literal translations.
> - A literal translation is where each word is translated to the corresponding word in the target language.
> - This approach presents some issues. For one case, there may not be an equivalent word in the target language.
> - Another case is where literal translation can change the meaning of the phrase or not get the context correct.
> - Artificial intelligence systems must be able to understand, not only the words, but also the semantic context in which they're used.
> - In this way, the service can return a more accurate translation of the input phrase or phrases.
> - The grammar rules, formal versus informal, and colloquialisms all need to be considered.

### **Text and speech translation**

- *Text translation* can be used to translate documents from one language to another.
- *Speech translation* is used to translate between spoken languages, sometimes directly (speech-to-speech translation) and sometimes by translating to an intermediary text format (speech-to-text translation).

## **Understand translation in Azure**

Microsoft provides Azure AI services that support translation. 

- The **Azure AI Translator** service, which supports text-to-text translation.
- The **Azure AI Speech** service, which enables speech to text and speech-to-speech translation.

### **Azure AI Translator**

- **Azure AI Translator →**  This service uses a Neural Machine Translation (NMT) model for translation, which analyzes the semantic context of the text and renders a more accurate and complete translation as a result.
- **Language support →**
    - Azure AI Translator supports text-to-text translation between more than 130 languages.
    - When using the service, you must specify the language you are translating ***from*** and the language you are translating ***to*** using ISO 639-1 language codes.
    - You can simultaneously translate a source document into multiple languages.

### **Azure AI Speech**

- You can use **Azure AI Speech** to translate spoken audio into the translation as text or an audio stream.
- This enables scenarios such as real-time closed captioning for a speech or simultaneous two-way translation of a spoken conversation.

## **Get started with translation in Azure**

- You can use **Azure AI Translator** or get **Azure AI Speech** with Speech Studio  with a programming language of your choice or the REST API.

## **Using Azure AI Translator**

Azure AI Translator includes the following capabilities:

- **Text translation** - used for quick and accurate text translation in real time across all supported languages.
- **Document translation** - used to translate multiple documents across all supported languages while preserving original document structure.
- **Custom translation** - used to enable enterprises, app developers, and language service providers to build customized neural machine translation (NMT) systems.

Azure AI Translator also offers some optional configuration to help you fine-tune the results:

- **Profanity filtering**. Without any configuration, the service will translate the input text, without filtering out profanity. Profanity levels are typically culture-specific but you can control profanity translation by either marking the translated text as profane or by omitting it in the results.
- **Selective translation**. You can tag content so that it isn't translated. For example, you may want to tag code, a brand name, or a word/phrase that doesn't make sense when localized.

### **Speech translation with Azure AI Speech**

Azure AI Speech includes the following capabilities:

- **Speech to text** - used to transcribe speech from an audio source to text format.
- **Text to speech** - used to generate spoken audio from a text source.
- **Speech Translation** - used to translate speech in one language to text or speech in another.

## **Exercise - Explore Azure AI Translator**

> 🔵 In Azure AI - Language Studio

---