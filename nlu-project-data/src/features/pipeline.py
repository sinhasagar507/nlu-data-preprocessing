from convokit.text_processing import TextProcessor, TextCleaner
from convokit.convokitPipeline import ConvokitPipeline
from heuristic_features import clean, convert_to_lower, sentiment_analyzer, modifier_count, \
    hedge_count
from heuristic_features import group_ref_count, subjectivity_utterance, measurePoliteness


def pipe():  # The Pipe would be expanded further
    clean_reddit_text = TextCleaner(text_cleaner=clean, replace_text=False, save_original=True)
    lowercase_text = TextProcessor(proc_fn=convert_to_lower, output_field="lowercase_text")
    analyze_sentiment = TextProcessor(proc_fn=sentiment_analyzer, input_field="lowercase_text",
                                      output_field="sentiment_polarity")
    subjectivity = TextProcessor(proc_fn=subjectivity_utterance, input_field="lowercase_text",
                                 output_field="subjectivity_score")
    modifier_count = TextProcessor(proc_fn=modifier_count, input_field="lowercase_text",
                                   output_field="modifier_count")
    hedge_modals = TextProcessor(proc_fn=hedge_count, input_field="lowercase_text",
                                 output_field="hedge_count")
    group_ref_count = TextProcessor(proc_fn=group_ref_count, input_field="lowercase_text",
                                    output_field="groupRef_count")
    politeness_markers = TextProcessor(proc_fn=measurePoliteness, input_field="lowercase_text",
                                       output_field="politeness_markers")
    metadata_pipe = ConvokitPipeline([
        ("convert to lowercase", lowercase_text),
        ("sentiment analyzer", analyze_sentiment),
        ("count modifiers", modifier_count),
        ("count hedges and modals", hedge_modals),
        ("count group references", group_ref_count),
        ("subjectivity score", subjectivity),
        ("politeness indicators", politeness_markers)
    ])

    return metadata_pipe
