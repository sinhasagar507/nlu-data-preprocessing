from convokit import Corpus
from n
from convokit.text_processing import TextProcessor, TextCleaner
from convokit.convokitPipeline import ConvokitPipeline
from src.data.make_dataset import clean
from build_heuristic_features import convert_to_lower, sentiment_analyzer, normalized_modifier_cnt, \
    normalized_hedge_cnt
from build_heuristic_features import subjectivity_score, normalized_group_ref_cnt, subjectivity_score, \
    measure_politeness


# The following pipe would be expanded further
clean_reddit_text = TextCleaner(text_cleaner=clean, replace_text=False, save_original=True)
lowercased_text = TextProcessor(proc_fn=convert_to_lower, output_field="lowercase_text")
analyze_sentiment = TextProcessor(proc_fn=sentiment_analyzer, input_field="lowercase_text",
                                  output_field="sentiment_polarity")
score_subjectivity = TextProcessor(proc_fn=subjectivity_score, input_field="lowercase_text", output_field="subjectivity_score")
modifier_count = TextProcessor(proc_fn=normalized_modifier_cnt, input_field="lowercase_text",
                               output_field="normalized_modifier_count")
hedge_modals = TextProcessor(proc_fn=normalized_hedge_cnt, input_field="lowercase_text",
                             output_field="normalized_hedge_cnt")
group_ref_count = TextProcessor(proc_fn=normalized_group_ref_cnt, input_field="lowercase_text",
                                output_field="normalized_group_ref_cnt")
politeness_markers = TextProcessor(proc_fn=measure_politeness, input_field="lowercase_text",
                                   output_field="politeness_markers")

pipe_reddit_uncleaned = ConvokitPipeline([
    ("convert to lowercase", lowercased_text),
    ("sentiment analyzer", analyze_sentiment),
    ("count modifiers", modifier_count),
    ("count hedges and modals", hedge_modals),
    ("count group references", group_ref_count),
    ("subjectivity score", subjectivity_score),
    ("politeness indicators", politeness_markers)
])

import pandas as pd
data = pd.read_csv("nlu-project-data/data/raw/submissions/sample_submission.csv")
# Load the corpus
corpus = Corpus(filename=)

# Finally, clean the reddit data
cleaned_corpus =


