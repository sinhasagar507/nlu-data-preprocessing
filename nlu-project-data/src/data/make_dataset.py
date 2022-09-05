# -*- coding: utf-8 -*-

# Primary Libraries
import re
import logging

import click
import contractions as cm

from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from pandas import DataFrame
from convokit import Corpus
from dotenv import find_dotenv, load_dotenv


def cleantext(
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


def createConversationGraph(graph: {}, child_list: List[str], parent_list: List[str]) -> Dict[str: List[str]]:
    """
    Creates a adjacency list of parent comment ids and child comment ids
    Args:
        graph: empty dictionary
        child_list: children comment ids
        parent_list: parent comment ids

    Returns:
        graph: conversation adjacency list

    """
    for child, parent in zip(child_list, parent_list):

        if parent not in graph.keys():
            graph[parent] = []
            graph[parent].append(child)
        else:
            graph[parent].append(child)

    return graph


def identify_bots(corpus: Corpus) -> List[str]:
    """
    Use a regex expression to search for explicit bots.

    Args :
      Conversation object

    Returns :
      List : a list of bots, if any from the conversation

    """

    remove = ["AutoModerator", "[deleted]"]
    blacklist = []
    speakers = corpus.get_speaker_ids()
    for speaker in speakers:
        result = re.search("\w+[^Ro|ro|][bB][oO][tT]$", speaker)
        if result:
            blacklist.append(speaker)

    blacklisted_speakers = remove + blacklist
    return blacklisted_speakers


def recurse_utterance_ids(graph: Dict[None: None], remove_utt_ids: List, iD: str) -> List:
    """
    Stores utterance IDs for all blacklisted speaker IDs  and their child speaker IDs

    Args:
        graph: conversation graph adjacency list
        remove_utt_ids: utterance ids with blacklisted speakers
        iD: utterance iD


    Returns:
        List of visited Ids that need to be removed from the utterances dataframe

    """
    remove_utt_ids.append(str(iD))

    if iD in list(graph.keys()):
        for idx in tqdm(graph[str(iD)]):
            recurse_utterance_ids(graph, remove_utt_ids, idx)
    else:
        return remove_utt_ids


def bot_removal(graph: Dict[str: List[str]], utt_df: DataFrame, blacklisted_speakers: List[str]) -> List[str]:
    """
    Role:
        Retrieve comment Ids of blacklisted speakers and its associated replies

    Args:
        graph: conversation graph adjacency list
        utt_df: utterances dataframe
        blacklisted_speakers: list of auto moderators, deleted utterance speakers and explicit bots

    Returns:
        List of utterance ids of blacklisted speakers and associated replies

    """
    remove_utt_ids = []
    for idx, row in utt_df.iterrows():
        if row["speaker"] in blacklisted_speakers:
            utt_ids = recurse_utterance_ids(graph, remove_utt_ids, idx)

    return utt_ids


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
