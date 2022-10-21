try:
    import os

    from convokit import Corpus
    from convokit.text_processing import TextProcessor, TextCleaner
    from convokit.convokitPipeline import ConvokitPipeline
    from ...src.data.make_dataset import cleantext
    from ...src.__init__ import ROOT_DIR
    from heuristics import convert_to_lower, measure_sentiment, normalized_modifier_cnt, \
        normalized_hedge_cnt, measure_subjectivity, normalized_group_ref_cnt, measure_politeness
except Exception as e:
    print(e)


def main():
    clean_text = TextCleaner(text_cleaner=cleantext, replace_text=False, save_original=True)
    lowercased_text = TextProcessor(proc_fn=convert_to_lower, output_field="lowercase_text")
    analyze_sentiment = TextProcessor(proc_fn=measure_sentiment, input_field="lowercase_text",
                                      output_field="sentiment_polarity")
    analyze_subjectivity = TextProcessor(proc_fn=measure_subjectivity, input_field="lowercase_text",
                                         output_field="subjectivity_score")
    modifiers = TextProcessor(proc_fn=normalized_modifier_cnt, input_field="lowercase_text",
                              output_field="normalized_modifier_count")
    hedges = TextProcessor(proc_fn=normalized_hedge_cnt, input_field="lowercase_text",
                           output_field="normalized_hedge_cnt")
    pronouns = TextProcessor(proc_fn=normalized_group_ref_cnt, input_field="lowercase_text",
                             output_field="normalized_group_ref_cnt")
    politeness = TextProcessor(proc_fn=measure_politeness, input_field="lowercase_text",
                               output_field="politeness_markers")

    discriminative_feature_pipe = ConvokitPipeline([  # The following pipe would be expanded further
        ("lowercase", lowercased_text),
        ("sentiment", analyze_sentiment),
        ("subjectivity", analyze_subjectivity)
        ("modifiers", modifiers),
        ("hedges", hedges)
        ("pronouns", pronouns),
        ("politeness", politeness),
    ])

    # Load the corpus
    sample_corpus = "sample_corpus"
    BASE_PATH = os.path.join(ROOT_DIR, "data", "interim", "unclean")
    corpus = Corpus(sample_corpus, filename=BASE_PATH)
    uncleaned_sample_corpus = discriminative_feature_pipe.transform(corpus)
    cleaned_sample_corpus = clean_text.transform(uncleaned_sample_corpus)

    # Dump Corpus here
    dump_dir = os.path.join(ROOT_DIR, "data", "interim", "interim", "clean")
    corpus.dump(cleaned_sample_corpus, dump_dir)


if __name__ == "__main__":
    main()
