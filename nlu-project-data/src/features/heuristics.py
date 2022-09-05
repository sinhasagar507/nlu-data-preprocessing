try:
    import os

    import warnings
    warnings.filterwarnings("ignore")
    import nltk
    from nltk.stem import WordNetLemmatizer
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from convokit import PolitenessStrategies
    import spacy
    import emot
    from textblob import TextBlob as tb
except Exception as e:
    print(e)


def convert_to_lower(utt: str) -> str:
    """ This function block performs twitter text normalization

        For e.g.,  the different forms of 'hate' are: Hate, HATE, haTE, etc. This function would convert all such occurrences to a single canonical form
        """

    exclude_tags_list = ['NN', 'NNS', 'NNP', 'NNPS']  # Check if the attached POS tags are correct or not
    modified_text_ls = []

    words = nltk.word_tokenize(utt)  # Tokenize the sentence and extract POS tags

    # lemmatizer = WordNetLemmatizer()  # Performs Lemmatization
    # words = [lemmatizer.lemmatize(word) for word in words]  # Perform lemmatization if required
    word_pos_tags = nltk.pos_tag(words)

    for (word, tag) in word_pos_tags:
        if tag not in exclude_tags_list:
            word = word.lower()
        modified_text_ls.append(word)

    utt = " ".join(modified_text_ls)

    return utt


def measure_sentiment(utt: str) -> dict:
    """

    Args:
        utt: Convokit utterance object

    Returns:
        A dictionary of negative, positive, neutral and composite sentiment scores for each utterance

    """
    sentiment = SentimentIntensityAnalyzer()  # Vader Sentiment Analysis
    sentence_ls = nltk.sent_tokenize(utt)
    sentiment_score_ls = []

    for sent in sentence_ls:
        sentiment_score_ls.append(sentiment.polarity_scores(sent))

    pos_score_sum, neu_score_sum, neg_score_sum = 0, 0, 0
    for sentiment_scores in sentiment_score_ls:
        pos_score_sum += sentiment_scores["pos"]
        neu_score_sum += sentiment_scores["neu"]
        neg_score_sum += sentiment_scores["neg"]

    pos_score_sum_avg = pos_score_sum / len(sentiment_score_ls)
    neu_score_sum_avg = neu_score_sum / len(sentiment_score_ls)
    neg_score_sum_avg = neg_score_sum / len(sentiment_score_ls)

    compound_sentiment_scores = {
        "pos": pos_score_sum_avg,
        "neu": neu_score_sum_avg,
        "neg": neg_score_sum_avg
    }
    return compound_sentiment_scores


def normalized_modifier_cnt(
        utt: str) -> float:  # Calculating less of something isn't always the best indicator. Instead the prevalence of something more than ususal is a better marker. # Optional - Emergency Toolkit
    """

    Args:
        utt: Convokit utterance object

    Returns:
        Modifier list AND normalized modifier count scores for each utterance

    """
    adj_pos_tags = ['JJ', 'JJR', 'JJS']  # POS tags describing adjectives
    adv_pos_tags = ['RB', 'RBR', 'RBS']  # POS tags for adverbs
    all_tags = adj_pos_tags + adv_pos_tags
    words = nltk.word_tokenize(utt)
    word_tag_lst = nltk.pos_tag(words)
    modifier_list = []
    cnt_tags = 0
    for (word, tag) in word_tag_lst:
        if tag in all_tags:
            modifier_list.append((word, tag))
            cnt_tags += 1
    normalized_tag_cnt = cnt_tags / len(utt)
    return normalized_tag_cnt, normalized_tag_cnt


def normalized_hedge_cnt(utt: str) -> float:
    """

    Args:
        utt: Convokit utterance object

    Returns:
        utt_hedge_modals: List of hedges in the utterance and normalized hedge count for each utterance
    """
    cnt_mods = 0
    set_of_hedges_en = ["almost", "apparent", "apparently", "appear", "appeared", "appears", "approximately", "argue",
                        "argued", "argues", "around", "assume", "assumed", "broadly", "certain amount",
                        "certain extent", "certain level", "claim", "claimed", "claims", "doubt", "doubtful",
                        "essentially", "estimate", "estimated", "fairly", "feel", "feels", "felt", "frequently",
                        "from my perspective", "from our perspective", "from this perspective", "generally", "guess",
                        "in general",
                        "in most cases", "in most instances", "in my opinion", "in my view", "in our opinion",
                        "in our view", "in this view", "indicate", "indicated", "indicates", "largely", "likely",
                        "mainly", "may",
                        "maybe", "might", "mostly", "often", "on the whole", "ought", "perhaps", "plausible",
                        "plausibly", "possible", "possibly", "postulate", "postulated", "postulates", "presumable",
                        "presumably", "probable",
                        "probably", "quite", "rather", "relatively", "roughly", "seems", "should", "sometimes",
                        "somewhat", "suggest", "suggested", "suggests", "suppose", "supposed", "supposes", "suspect",
                        "suspects", "tend to",
                        "tended to", "tends to", "think", "thinking", "thought", "to my knowledge", "typical",
                        "typically", "uncertain",
                        "uncertainly", "unclear", "unclearly", "unlikely",
                        "usually"]  # The Hedge word list has been taken from "https://github.com/tslmy/politeness-estimator.git"

    pos_modal_ls = ["shall", "should", "can", "could", "will", "would", "may", "must",
                    "might"]  # List of 9 modal verbs indicating possibility

    hedges_modals = set_of_hedges_en + pos_modal_ls

    words = utt.lower().split()
    utt_hedge_modals = []
    for word in words:
        if word in hedges_modals:
            utt_hedge_modals.append(word)
            cnt_mods += 1

    normalized_hedge_cnt = cnt_mods / len(utt)
    return utt_hedge_modals, normalized_hedge_cnt


def normalized_group_ref_cnt(utt: str) -> int:  # Include all third-party pronouns as well
    """

    Args:
        utt: Convokit utterance object

    Returns:
       List of normalized group reference and normalized group count

    """
    cnt_group_ref = 0
    words = utt.lower().split()
    group_ref = ["he", "him", "his", "himself", "she", "her", "hers", "herself", "you", "yours", "yourself",
                 "yourselves", "it", "its", "itself",
                 "we", "our", "ours", "ourselves", "us", "they", "them", "themselves", "their", "theirs", "everyone",
                 "everybody"]
    utt_group_ref = []
    for word in words:
        if word in group_ref:
            utt_group_ref.append(word)
            cnt_group_ref += 1
    return utt_group_ref, cnt_group_ref


def measure_subjectivity(utt: str) -> dict:
    """

    Args:
        utt: Convokit utterance object

    Returns:
        Returns subjectivity scores for a Convokit utterance

    """
    subjective_lexicon_cnt, subjectivity_scores = 0, []
    sents = nltk.sent_tokenize(utt)
    subjectivity_clues = []

    filename = os.path.join(os.getcwd(), "data", "external", "subjectivity_clues.txt")
    with open(filename, "r") as file:
        for line in file.readlines():
            values = line.split(" ")[2]
            subjectivity_clue = values.split("=")[1]
            subjectivity_clues.append(subjectivity_clue)
    for sent in sents:
        subjectivity_scores.append(tb(sent).sentiment.subjectivity)
        words = sent.lower().split()

        for word in words:
            if word in subjectivity_clues:
                subjective_lexicon_cnt += 1

    avg_subjectivity_score = sum(subjectivity_scores) / len(subjectivity_scores)
    subjective_details = {
        "avg_subjectivity_score": avg_subjectivity_score,
        "normalized_subjective_lexicon_cnt": subjective_lexicon_cnt / len(subjectivity_scores)
    }

    return subjective_details


def measure_politeness(utt: str):
    """
    Args:
        utt: Convokit utterance object

    Returns:
        Returns values for politeness strategy parameters for the utterance object, as defined in the https://aclanthology.org/P13-1025.pdf
    Returns:

    """
    politeness = PolitenessStrategies()  # Politeness Indicators
    spacy_nlp = spacy.load('en_core_web_md', disable=["ner"])  # Spacy Language Model
    utt = politeness.transform_utterance(utt, spacy_nlp=spacy_nlp)
    return utt.meta["politeness_strategies"]


def cntEmojis(utterance: str) -> int:  # Optional - Emergency Toolkit -
    """ Returns emoji list and its length in an utterance """
    emotion = emot.core.emot()  # Initialize Emoji object
    emot_dict = emotion.emoji(utterance)

    return emot_dict['value'], len(emot_dict['value'])
