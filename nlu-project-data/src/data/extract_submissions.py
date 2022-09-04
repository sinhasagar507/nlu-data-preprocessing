try:
    import argparse as ap
    import csv
    import datetime
    import json
    from json import JSONDecodeError
    import os
    import requests
    import time

    from tqdm import tqdm
except Exception as e:
    print(e)


def extract_data(submission):
    submission_data = []
    title = submission["title"]

    try:
        flair = submission["link_flair_text"]  # flair is not always present so we wrap it in try/except
    except KeyError:
        flair = "NaN"

    submission_id = submission["id"]
    score = submission["score"]
    created = datetime.datetime.fromtimestamp(submission['created_utc'])  # 1520561700.0
    num_comms = submission["num_comments"]
    permalink = submission["permalink"]

    try:
        text = submission["selftext"]
    except KeyError:
        text = "NaN"

    submission_data.append((submission_id, title, score, created, num_comms, permalink, flair, text))

    return submission_id, submission_data


def get_PSAW_data(**kwargs):
    submission_endpoint = "https://api.pushshift.io/reddit/search/submission/?size=1000&after=" + str(
        kwargs["after"]) + "&before=" + str(kwargs["before"]) + "&subreddit=" + str(
        kwargs["sub"]) + "&num_comments=>60" + "&over_18" + str(kwargs["over_18"])
    print(f"The submission endpoint/URL : {submission_endpoint}")
    submissions = requests.get(submission_endpoint)
    data = json.loads(submissions.text)
    return data["data"]


def save_file(filename, submission_stats):
    submission_dir = os.path.join(os.getcwd(), "data", "raw", "submissions")
    file = os.path.join(submission_dir, filename)
    with open(file, "w", newline="", encoding="utf-8") as file:
        upload_count = 0
        a = csv.writer(file, delimiter=",")
        headers = ["sub_id", "title", "score", "created", "num_comms", "permalink", "flair", "text"]
        a.writerow(headers)
        for submission in submission_stats:
            a.writerow(submission_stats[submission][0])
            upload_count += 1
        print(str(upload_count) + " submissions have been uploaded")


def main():
    parser = ap.ArgumentParser()
    parser.add_argument("-bd", "--begin_date", help="extract submissions on and after the specified date",
                        type=int)  # 1st date of a month as 1. Throw exception for out-of-bound dates
    parser.add_argument("-ed", "--end_date", help="extract submissions on and before the specified date",
                        type=int)  # last date of a month as 30 or 31. Throw exception for out-of-bound dates
    parser.add_argument("-bm", "--begin_month", help="extract submissions beginning the specified month",
                        type=int)  # first month as 01. Throw exception as specified for the date
    parser.add_argument("-em", "--end_month", help="extract submissions up till the specified month",
                        type=int)  # last month as 12. Throw exception as specified for the date
    parser.add_argument("-y", "--year", help="extract subs for the specified year",
                        type=int)  # Enter the year span between 2018 AND 2022. Extend to include a greater range later
    parser.add_argument("-sub", "--subreddit", help="extract submissions from the specified subreddit",
                        type=str)  # subreddit for which the data has to be extracted
    parser.add_argument("-comms", "--comments", help="minimum number of comments per submission",
                        type=int)  # Optional argument. By default, the explicit flag is set to 1
    parser.add_argument("--over_18",
                        help="include explicit submissions",
                        action="store_true")
    parser.add_argument("-f", "--filename",
                        help="CSV file for storing submissions",
                        type=str)

    args = parser.parse_args()

    begin_date = args.begin_date
    end_date = args.end_date
    begin_month = args.begin_month
    end_month = args.end_month
    year = args.year

    start = f"{begin_date}-{begin_month}-{year}"
    end = f"{end_date}-{end_month}-{year}"
    start_timestamp = int(time.mktime(datetime.datetime.strptime(start, "%d-%m-%Y").timetuple()))
    end_timestamp = int(time.mktime(datetime.datetime.strptime(end, "%d-%m-%Y").timetuple()))

    subreddit = args.subreddit
    num_comms = args.comments

    over_18 = args.over_18
    filename = args.filename

    submission_count = 0
    submission_stats = {}

    data = get_PSAW_data(after=start_timestamp, before=end_timestamp, sub=subreddit, comms=num_comms, over_18=over_18)

    try:
        while len(data) > 0:

            for submission in tqdm(data):
                submission_id, submission_data = extract_data(submission)
                submission_count += 1
                submission_stats[submission_id] = submission_data

            print(len(data))
            print(str(datetime.datetime.fromtimestamp(data[-1]["created_utc"])))

            try:
                start = str(data[-1]["created_utc"])
                data = get_PSAW_data(after=start, before=end_timestamp, sub=subreddit, comms=num_comms,
                                     over_18=over_18)

            except JSONDecodeError:
                time.sleep(5)
                start = str(data[-1]["created_utc"])
                data = get_PSAW_data(after=start, before=end_timestamp, sub=subreddit, comms=num_comms,
                                     over_18=over_18)

    except KeyboardInterrupt:
        save_file(filename, submission_stats)

    else:
        save_file(filename, submission_stats)


if __name__ == "__main__":
    main()
