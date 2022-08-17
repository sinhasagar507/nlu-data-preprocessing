import re
import os
import json

import warnings

warnings.filterwarnings("ignore")

import nltk
from nltk.stem import WordNetLemmatizer

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from convokit import Corpus, PolitenessStrategies
from convokit.text_processing import TextProcessor, TextCleaner, TextParser
import spacy
import contractions as cm
from textblob import TextBlob


lemmatizer = WordNetLemmatizer()  # Performs Lemmatization
sentiment = SentimentIntensityAnalyzer()  # Vader Sentiment Analysis
politeness = PolitenessStrategies()  # Politeness Indicators
spacy_nlp = spacy.load('en_core_web_md', disable=["ner"])  # Spacy Language Model


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

        For eg.,  the different forms of 'hate' are: Hate, HATE, haTE, etc. This function would convert all such occurences to a single canonical form
        """

    exclude_tags_list = ['NN', 'NNS', 'NNP', 'NNPS']  # Check if the attached POS tags are correct or not
    modified_text_ls = []

    words = nltk.word_tokenize(utt)  # Tokenize the sentence and extract POS tags

    words = [lemmatizer.lemmatize(word) for word in words]  # Perform lemmatization if required
    word_pos_tags = nltk.pos_tag(words)

    for (word, tag) in word_pos_tags:
        if tag not in exclude_tags_list:
            word = word.lower()
        modified_text_ls.append(word)

    utt = " ".join(modified_text_ls)

    return utt


# Use a combination of IDENTITY ATTACK and INSULT parameters to separate MICROAGGRESSIONS from OTHER HATE-SPEECH forms
def sentiment_analyzer(utt: str) -> dict:
    return sentiment.polarity_scores(utt)


def normalized_modifier_cnt(
        utt: str) -> float:  # Calculating less of something isn't always the best indicator. Instead the prevalence of something more than ususal is a better marker. # Optional - Emergency Toolkit
    """

    Args:
        utt:

    Returns:


    """
    adj_pos_tags = ['JJ', 'JJR', 'JJS']  # POS tags describing adjectives
    adv_pos_tags = ['RB', 'RBR', 'RBS']  # POS tags for adverbs
    words = nltk.word_tokenize(utt)
    word_tag_lst = nltk.pos_tag(words)
    cnt_tags = 0
    for (word, tag) in word_tag_lst:
        if tag in adj_pos_tags or tag in adv_pos_tags:
            cnt_tags += 1

    return cnt_tags / len(utt)


def normalized_hedge_cnt(utt: str) -> float:
    """

    Args:

    Returns:

    """
    cnt_mods = 0
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
            cnt_mods += 1
    return cnt_mods / len(utt)


def normalized_group_ref_cnt(utt: str) -> int:  # Include all third-party pronouns as well
    """

    Args:

    Returns:

    """
    cnt_group_ref = 0
    words = utt.lower().split()
    group_ref = ["we", "our", "ours", "ourselves", "us", "they", "them", "thesmselves", "their", "theirs",
                 "everyone", "everybody"]  # More of it to be included here. Self-referencing pronouns

    for word in words:
        if word in group_ref:
            cnt_group_ref += 1
    return cnt_group_ref / len(utt)


def subjectivity_score(utt: str) -> int:
    """

    Args:

    Returns:

    """
    return TextBlob(utt).sentiment.subjectivity


def measure_politeness(utt: str):
    """
    Args:

    Returns:

    """
    utt = politeness.transform_utterance(utt, spacy_nlp=spacy_nlp)
    return utt.meta["politeness_strategies"]

