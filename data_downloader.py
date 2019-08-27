from math import ceil
import datetime as dt
import requests
import json

from typing import Optional, List, Dict, Iterable, Tuple, Collection

import numpy as np

from tqdm import tqdm

import pandas as pd


def chunker(seq: Collection, size: int) -> Iterable:
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def get_posts_between(start: int, end: int, field_names: Optional[List[str]] = None) -> pd.DataFrame:
    if field_names is None:
        field_names = ['id']  # , 'title', 'selftext', 'score', 'author']

    # TODO: can be optimized by avoiding pandas (probably not worth it)
    api_url = 'https://api.pushshift.io/reddit/submission/search'

    req = requests.get(api_url,
                       params={
                           "subreddit": "AmITheAsshole",
                           "sort": "desc",
                           "sort_type": "score",
                           "after": start,
                           "before": end,
                           "limit": 10000
                       })

    res = json.loads(req.content)

    filtered = filter(
        lambda x: all(name in x.keys() for name in field_names),
        res['data']
    )

    data = list(map(
        lambda x: tuple(x[field] for field in field_names),
        filtered
    ))

    df = pd.DataFrame(data, columns=field_names)
    df['name'] = 't3_' + df['id']

    return df


def request_info(names: List[str], useragent: str = 'MoralJudgementIncoming') -> List[Dict]:
    all_names = ','.join(names)

    res = requests.get('https://www.reddit.com/api/info.json',
                       params={'id': all_names},
                       headers={'User-agent': useragent})

    posts_info = json.loads(res.content)['data']['children']

    #     print(len(posts_info), len(names))
    assert len(posts_info) == len(names), ValueError("Didn't get all posts' info")

    return posts_info


def get_details(names: Collection[str]) -> pd.DataFrame:
    # TODO: add error handling if a chunk fails
    chunk_size = 100

    name_chunks = chunker(names, chunk_size)

    all_info = []

    chunk_count = ceil(len(names) / chunk_size)

    for chunk in tqdm(name_chunks, total=chunk_count):
        info = request_info(chunk)

        all_info.extend(info)

    field_names = ['id', 'title', 'created', 'selftext', 'score', 'edited', 'link_flair_text']

    data = list(map(
        lambda x: (x['data'][field] for field in field_names),
        all_info
    ))

    df_new = pd.DataFrame(data, columns=field_names)
    df_new = df_new.fillna('')
    df_new = df_new.set_index('id')

    return df_new


def iterate_days(start: Tuple[int, int, int],
                 end: Tuple[int, int, int],
                 diff: int = 24,
                 finer_start: Optional[Tuple[int, int, int]] = None) -> Tuple[dt.datetime, dt.datetime]:
    """

    Args:
        start: (year, month, day), start date
        end: (year, month, day), end date
        diff: number of hours to iterate over
        finer_start: (year, month, day), optional time to reduce the diff by half, for better granularity
    """
    start_dt = dt.datetime(*start)
    end_dt = dt.datetime(*end)

    assert start_dt < end_dt, ValueError("Start date should be before end date")

    current_dt = dt.datetime(*start)

    if finer_start is None:
        finer = True
        finer_dt = None
    else:
        finer = False
        finer_dt = dt.datetime(*finer_start)

    while current_dt != end_dt:
        start = current_dt
        end = current_dt + dt.timedelta(hours=diff)

        # If we're not in fine mode yet, and we passed a certain date, decrease the diff

        if not finer and current_dt >= finer_dt:
            finer = True
            diff = diff // 2

        current_dt = end
        yield int(start.timestamp()), int(end.timestamp())


def get_all_data(start: Tuple[int, int, int],
                 end: Tuple[int, int, int],
                 diff: int = 24,
                 finer_start: Optional[Tuple[int, int, int]] = None,
                 save_names: Optional[str] = None,
                 save_data: Optional[str] = None) -> pd.DataFrame:

    days = iterate_days(start, end, diff, finer_start)

    df_list = []

    num_days = (dt.datetime(*end) - dt.datetime(*start)).days

    if finer_start is not None:
        num_days += (dt.datetime(*end) - dt.datetime(*finer_start)).days

    print("Downloading post names from pushshift API")

    for start, end in tqdm(days, total=num_days):
        df_part = get_posts_between(start=start, end=end)
        df_list.append(df_part)

    #     plt.plot(list(map(lambda x: x.name.shape[0], df_list)))

    all_names = np.concatenate(list(map(lambda df: df.name.values, df_list)))

    if save_names is not None:
        with open(save_names, 'w') as f:
            for name in all_names:
                f.write(name)

    print("Downloading post details from reddit API")

    df_full = get_details(all_names)

    if save_data is not None:
        df_full.to_csv(save_data)

    return df_full


def obtain_default_data(path: Optional[str] = None, save_path: Optional[str] = None):
    if path is None:
        df_full = get_all_data(start=(2018, 9, 1),
                               end=(2019, 8, 20),
                               diff=24,
                               finer_start=(2019, 5, 1),
                               save_data=save_path
                               )
        return df_full
    else:
        return pd.read_csv(path)


