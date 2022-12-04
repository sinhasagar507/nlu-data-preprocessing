try:
    import os
    import json
    from typing import Dict
    from convokit import Utterance, Corpus
    from googleapiclient import discovery
    from tqdm import tqdm

except Exception as e:
    print(e)


def hate_microaggression_polarity(utt: Utterance) -> Dict:
    """
    Calculates probability scores for production attributes of Perspective API
    A detailed description can be obtained at "https://developers.perspectiveapi.com/s/about-the-api-attributes-and-languages"


    Args:
        utt: Convokit utterance object

    Returns:
        Probability scores for the parameters

    """

    developer_key = os.environ.get("GOOGLE_API_CLIENT_INSTITUTION")  # load up the entries as environment variables

    client = discovery.build(  # Initialize the client
        "commentanalyzer",
        "v1alpha1",
        developerKey=developer_key,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
    )

    analyze_request = {
        'comment': {'text': utt},
        'requestedAttributes': {
            'TOXICITY': {},
            'SEVERE_TOXICITY': {},
            'IDENTITY_ATTACK': {},
            'INSULT': {},
            'PROFANITY': {}
        }
    }

    response = client.comments().analyze(body=analyze_request).execute()
    response_dict = dict(response)

    attributes = ["toxicity", "severe_toxicity", "identity_attack", "insult", "profanity"]
    attribute_values = {}

    for attr in attributes:
        attribute_values[attr] = response_dict["attributeScores"][attr]["spanScores"][0]["score"]["value"]

    return attribute_values
