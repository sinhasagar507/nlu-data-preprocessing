try:
    import os
    from ...src.__init__ import ROOT_DIR

    from convokit import Corpus, TextCleaner
    from make_dataset import bot_removal, cleantext, createConversationGraph, identify_bots, recurse_utterance_ids

except Exception as e:
    print(e)


def main():
    file = os.path.join(ROOT_DIR, "data", "interim", "unclean", "sample_corpus")  # Load the Corpus
    corpus = Corpus(filename=file)

    clean = TextCleaner(text_cleaner=cleantext, replace_text=False, save_original=True)
    corpus = clean.transform(corpus)

    utt_df = corpus.get_utterances_dataframe().drop("vectors", axis=1)
    child_list = list(utt_df.index)
    parent_list = list(utt_df["reply_to"])

    graph = {}
    graph = createConversationGraph(graph, child_list, parent_list)
    bots = identify_bots(corpus)
    remove_utt_id_ls = bot_removal(graph, utt_df, bots)
    utt_df.drop(remove_utt_id_ls, inplace=True)

    # Create the new corpus and dump it
    processed_sample_corpus = Corpus.from_pandas(utt_df)
    dump_dir = os.path.join(ROOT_DIR, "data", "interim", "cleaned")
    corpus.dump(processed_sample_corpus, dump_dir)


if "__name__" == "__main__":
    main()
