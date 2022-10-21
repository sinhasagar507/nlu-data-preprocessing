try:
    import os
    import time
    import datetime
    import pandas as pd
    from tqdm import tqdm
    import praw
except Exception as e:
    print(e)

reddit_client_id = os.environ.get("SAGAR_REDDIT_CLIENT_ID")  # load up the variable value in environment variables
reddit_client_secret = os.environ.get("SAGAR_REDDIT_CLIENT_SECRET")
reddit_password = os.environ.get("SAGAR_REDDIT_PASSWORD")
reddit_username = os.environ.get("SAGAR_REDDIT_USERNAME")

reddit = praw.Reddit(
    client_id=reddit_client_id,
    client_secret=reddit_client_secret,
    password=reddit_password,
    username=reddit_username,
    user_agent="personal use script",
    check_for_async=False
)


def readSubmissionIds(filepath):
    """
    Stores submission id and its associated metadata

    Args:
        filepath: path to raw data file

    Returns:
        A dictionary of submission IDs and corresponding meta-data such as speaker name, timestamp, score, no. of comments and title text
    """
    data = pd.read_csv(str(filepath))
    data.columns = data.columns.str.replace(" ", "")
    data = data[data["num_comms"] > 0]

    submission_id_list, submission_metadata = [], {}

    for idx in tqdm(data.index):
        submission_id = data.loc[idx, "post_id"]
        submission_id_list.append(submission_id)

        speaker = data.loc[idx, "speaker"]
        score = data.loc[idx, "score"]

        timestamp = data.loc[idx, "timestamp"]
        timestamp = int(time.mktime(datetime.strptime(timestamp, '%d-%m-%Y %H:%M').timetuple()))

        num_comms = data.loc[idx, "num_comms"]
        text = data.loc[idx, "text"]

        submission_metadata[str(submission_id)] = [(speaker, timestamp, score, num_comms, text)]

    return submission_id_list, submission_metadata
