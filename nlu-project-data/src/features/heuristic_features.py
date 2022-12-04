try:
    import re
    import os
    from collections import defaultdict

    import warnings
    warnings.filterwarnings("ignore")
    import emot
    import nltk
    from nltk.stem import WordNetLemmatizer
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from convokit import Corpus, PolitenessStrategies
    from convokit.text_processing import TextProcessor, TextCleaner, TextParser
    import spacy
    import contractions as cm
    from textblob import TextBlob as tb

    from ...config import config

except Exception as e:
    print(e)


def def_value():
    return 0


def subjective_words():
    subjectivity_clues = []
    with open(os.path.join(config.ROOT_DIR, "data", "external", "subjectivity_clues"), "r") as f:
        for line in f.readlines():
            values = line.split(" ")[2]
            subjectivity_clues = values.split("=")[1]
            subjectivity_clues.append(subjectivity_clues)
    return subjectivity_clues


def clean(
        text, newline=True, quote=True,
        bullet_point=True, dates=True, link=True,
        strikethrough=True, spoiler=True, heading=True,
        emoj=True, emoticon=True, condensed=True):
    # Newlines we don't need - only
    """ Cleans reddit utterances"""

    if newline:
        text = re.sub(r'\n+', ' ', text)
        # Remove the many " " that we replaced in the last step
        text = text.strip()
        text = re.sub(r'\s\s+', ' ', text)

    # > are for the quoted texts from the main comment or the reply
    if quote:
        text = re.sub(r'>', '', text)

    # Bullet points/asterisk are used for markdown like - bold/italic - Could create trouble in parsing? idk
    if bullet_point:
        text = re.sub(r'\*', '', text)
        text = re.sub('&amp;#x200B;', '', text)

    # []() Link format then we remove both the tag/placeholder and the link
    if link:
        text = re.sub(r"http\S+", '', text)
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)

    # Strikethrough
    if strikethrough:
        text = re.sub('~', '', text)

    # Spoiler, which is used with < less-than (Preserves the text)
    if spoiler:
        text = re.sub('&lt;', '', text)
        text = re.sub(r'!(.*?)!', r'\1', text)

    # Heading to be removed as there are these markdown style features in reddit too
    if heading:
        text = re.sub('#', '', text)

    if emoj:
        # Implement the emoji scheme here
        # Makes more sense for the node feature but might as well import that function here if ready
        # Implementing a Naive Emoji Scheme
        # Some associated libraries are EMOT and DEMOJI
        # text = emoji.demojize(text).replace(":", "").replace("_", "")
        pass

    if dates:
        text = re.sub(r'(\d+/\d+/\d+)', '', text)

    if emoticon:
        # Implement the emoticon scheme here.
        # Makes more sense for the node feature but might as well import that function here if ready
        pass

    # Needs to be the last step in the process
    if condensed:
        text = cm.fix(text)
        # print("Running")

    return text


def convert_to_lower(utt: str) -> str:
    """ This function block performs twitter text normalization

        For instance, the different forms of 'hate' are: Hate, HATE, haTE, etc. This function would convert all such occurences to a single canonical form
        """
    exclude_tags_list = ['NN', 'NNS', 'NNP', 'NNPS']  # Check if the attached POS tags are correct or not
    sents = nltk.sent_tokenize(utt)
    modified_sent_ls = []

    for sent in sents:
        modified_token_ls = []
        words = nltk.word_tokenize(sent)  # Tokenize the sentence and extract POS tags
        lemmatizer = WordNetLemmatizer()  # Performs Lemmatization
        words = [lemmatizer.lemmatize(word) for word in words]  # Perform lemmatization if required
        word_pos_tags = nltk.pos_tag(words)

        for (word, tag) in word_pos_tags:
            if tag not in exclude_tags_list or word != "I":
                word = word.lower()
                modified_token_ls.append(word)

        modified_token_ls[0] = modified_token_ls[0].capitalize()

        utt = " ".join(modified_token_ls)
        utt = utt.strip()
        modified_sent_ls.append(sent)

    final_text = " ".join(modified_sent_ls)

    return final_text


# Use a combination of IDENTITY ATTACK and INSULT parameters to separate MICROAGGRESSIONS from OTHER HATE-SPEECH forms
def sentiment_analyzer(utt: str) -> dict:
    sentiment = SentimentIntensityAnalyzer()  # Intialize Vader Sentiment Analyzer
    sentence_ls = nltk.sent_tokenize(utt)
    sentiment_score_ls = []

    for sent in sentence_ls:
        sentiment_score_ls.append(sentiment.polarity_scores(sent))

    pos_score_sum, neu_score_sum, neg_score_sum = 0, 0, 0
    for sentiment_scores in sentiment_score_ls:
        pos_score_sum += sentiment_scores["pos"]
        neu_score_sum += sentiment_scores["neu"]
        neg_score_sum += sentiment_scores["neg"]

    pos_score_sum_avg = round((pos_score_sum / len(sentiment_score_ls)), 3)
    neu_score_sum_avg = round((neu_score_sum / len(sentiment_score_ls)), 3)
    neg_score_sum_avg = round((neg_score_sum / len(sentiment_score_ls)), 3)

    compound_sentiment_scores = {
        "pos": pos_score_sum_avg,
        "neu": neu_score_sum_avg,
        "neg": neg_score_sum_avg
    }
    return compound_sentiment_scores


analyze_sentiment = TextProcessor(proc_fn=sentiment_analyzer, input_field="lowercase_text",
                                  output_field="sentiment_polarity")


def modifier_count(utt: str) -> int:  # Calculating less of something isn't always the best indicator. Instead the prevalence of something more than ususal is a better marker. # Optional - Emergency Toolkit
    """Count modifiers, i.e., adjectives and adverbs in an utterance
    Practically every sentence has modifiers. This function doesn't act as a filter. It is intended to be applied to the entire dataframe
    The function block can detect probable deceptive clues in tweets and reddit posts
    Less usage of descriptive modifiers is a possible clue that the speaker is uncertain in his claims/opinions.
    """

    adj_pos_tags = ['JJ', 'JJR', 'JJS']  # POS tags describing adjectives
    adv_pos_tags = ['RB', 'RBR', 'RBS']  # POS tags for adverbs
    words = nltk.word_tokenize(utt)
    word_tag_lst = nltk.pos_tag(words)
    mod_count_dict = defaultdict(def_value)
    count_mod_tags = 0
    for (word, tag) in word_tag_lst:
        if tag in adj_pos_tags or tag in adv_pos_tags:
            count_mod_tags += 1
            mod_count_dict[word] += 1
    return {"modifier_count_dict": mod_count_dict, "count_mod_tags": count_mod_tags}


def hedge_count(utt: str) -> int:
    """ Count the list of all modal verbs that indicate possibility, but not certainty
    The function block can detect probable deceptive clues in tweets and reddit posts
    More usage of uncertain modal verbs is a possible clue that the speaker is uncertain in his utterance
    """
    count_hedges, hedge_count_dict = 0, defaultdict(def_value)
    set_of_hedges_en = ["almost", "apparent", "apparently", "appear", "appeared", "appears", "approximately", "argue",
                        "argued", "argues", "around", "assume", "assumed", "broadly", "certain amount",
                        "certain extent", "certain level", "claim", "claimed", "claims", "doubt", "doubtful",
                        "essentially", "estimate", "estimated", "fairly", "feel", "feels", "felt", "frequently",
                        "from my perspective",
                        "from our perspective", "from this perspective", "generally", "guess", "in general",
                        "in most cases", "in most instances", "in my opinion", "in my view", "in our opinion",
                        "in our view",
                        "in this view", "indicate", "indicated", "indicates", "largely", "likely", "mainly", "may",
                        "maybe", "might", "mostly", "often", "on the whole", "ought", "perhaps", "plausible",
                        "plausibly", "possible",
                        "possibly", "postulate", "postulated", "postulates", "presumable", "presumably", "probable",
                        "probably", "quite", "rather", "relatively", "roughly", "seems", "should", "sometimes",
                        "somewhat", "suggest",
                        "suggested", "suggests", "suppose", "supposed", "supposes", "suspect", "suspects", "tend to",
                        "tended to", "tends to", "think", "thinking", "thought", "to my knowledge", "typical",
                        "typically", "uncertain",
                        "uncertainly", "unclear", "unclearly", "unlikely",
                        "usually"]  # The Hedge word list has been taken from "https://github.com/tslmy/politeness-estimator.git"

    pos_modal_ls = ["shall", "should", "can", "could", "will", "would", "may", "must",
                    "might"]  # List of 9 modal verbs indicating possibility

    hedges_modals = set_of_hedges_en + pos_modal_ls

    words = utt.lower().split(" ")
    for word in words:
        if word in hedges_modals:
            count_hedges += 1
            hedge_count_dict[word] += 1
    return {"hedge_count_dict": hedge_count_dict, "count_hedges": count_hedges}


def group_ref_count(utt: str) -> int:  # Include all third-party pronouns as well
    """Count list of group references
    Usage of more self-references along with subjectivity score is a possible indication of deception
    """
    count_group_ref, group_ref_count_dict = 0, defaultdict(def_value)
    words = utt.lower().split()
    group_ref = ["we", "our", "ours", "ourselves", "us", "they", "them", "thesmselves", "their", "theirs",
                 "everyone", "everybody"]  # More of it to be included here. Self-referencing pronouns

    for word in words:
        if word in group_ref:
            count_group_ref += 1
            group_ref_count_dict[word] += 1
    return {"group_ref_count_dict": group_ref_count_dict, "count_group_ref": count_group_ref}


def subjectivity_utterance(utt: str) -> int:
    """ Textblob subjectivity score
    A higher subjective score indicates personal opinion.
    Low subjective scores could be a possible indicator of deception. To be used along with self references.
    """
    subjective_lexicon_count, subjective_lexicon_dict, subjectivity_scores = 0, defaultdict(def_value), []
    sents = nltk.sent_tokenize(utt)

    for sent in sents:
        subjectivity_scores.append(tb(sent).sentiment.subjectivity)
        words = sent.lower().split()

        for word in words:
            subjectivity_clues = subjective_words()
            if word in subjectivity_clues:
                subjective_lexicon_dict[word] += 1
                subjective_lexicon_count += 1

    avg_subjectivity_score = round((sum(subjectivity_scores) / len(subjectivity_scores)), 3)
    subjective_details = {
        "avg_subjectivity_score": avg_subjectivity_score,
        "subjective_lexicon_count": subjective_lexicon_count,
        "subjective_lexicon_dict": subjective_lexicon_dict
    }

    return subjective_details


def measurePoliteness(utt: str):
    """
    Computes politeness indicators in the text. The 9 positive politeness strategies
    """
    politeness = PolitenessStrategies()  # Politeness Indicators
    spacy_nlp = spacy.load('en_core_web_md', disable=["ner"])  # Spacy Language Model
    transformed_utt = politeness.transform_utterance(utt, spacy_nlp=spacy_nlp)
    return transformed_utt.meta['politeness_strategies']


def count_emojis(utt: str) -> int:  # Optional - Emergency Toolkit -
    """ Counts the total number of emojis in an utterance
    We don't intend to delete tweets that have emojis. This function doesn't act as a filter. It is intended to be applied to the entire dataframe
    Decide if it is redundant or not - maybe some possible indicators - not the first choice anyhow
    """
    emotion = emot.core.emot()  # Initialize Emoji Object
    emot_dict = emotion.emoji(utt)

    return len(emot_dict['value'])
